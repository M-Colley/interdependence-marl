"""Dual-Head Recurrent CTDE Actor-Critic with Online Theory of Mind & Cross-Modal Grounding.

Integrates:
  - Egocentric entity-structured encoder over decentralized partial observations
  - Decentralized GRU history encoders tracking persistent beliefs and partner behavior
  - Online Theory-of-Mind (ToM) head inferring partner capabilities `phi_{-i}` and target
    beliefs `b_i` from behavioral cues (inspired by MToMnet, Bortoletto & Bulling, ECAI '24)
  - Cross-modal object-attention grounding head (inspired by MATE, Benintendi & Bulling, '26)
  - Dual action heads for Required Ability (`a_req`) and Supportive Ability (`a_supp`)
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from interdependence_marl.envs.adc_posg import (
    NUM_REQ_ACTIONS,
    NUM_STATIONS,
    NUM_SUPP_ACTIONS,
    OBS_DIM,
    STATE_DIM,
)
from interdependence_marl.profiles import ABILITY_DIM


@dataclass
class AgentStepOutput:
    a_req: torch.Tensor               # (B, 2)
    a_supp: torch.Tensor              # (B, 2)
    logp_req: torch.Tensor            # (B, 2)
    logp_supp: torch.Tensor           # (B, 2)
    entropy_joint: torch.Tensor       # (B, 2)
    value: torch.Tensor               # (B,)
    h_next: torch.Tensor              # (B, 2, H)
    pred_partner_phi: torch.Tensor    # (B, 2, 4)
    pred_target_logits: torch.Tensor  # (B, 2, 3)
    attn_probs: torch.Tensor          # (B, 2, 3)
    ground_emb: torch.Tensor          # (B, 2, 32)


def _sample_and_logp(
    logits: torch.Tensor, deterministic: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    if deterministic:
        acts = logits.argmax(dim=-1)
    else:
        acts = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(logits.shape[:-1])
    act_logp = log_probs.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
    ent = -(probs * log_probs).sum(dim=-1)
    return acts, act_logp, ent


def _eval_logp_and_ent(
    logits: torch.Tensor, acts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    act_logp = log_probs.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
    ent = -(probs * log_probs).sum(dim=-1)
    return act_logp, ent


def extract_egocentric_features(obs: torch.Tensor) -> torch.Tensor:
    """Compute strictly decentralized egocentric relational features from `obs (..., 42)`."""
    lead = obs.shape[:-1]
    own_pos = obs[..., 0:2]                                  # (..., 2)
    st_coords = obs[..., 8:14].reshape(*lead, 3, 2)          # (..., 3, 2)
    partner_pos = obs[..., 14:16]                            # (..., 2)
    vis_codes = obs[..., 16:25].reshape(*lead, 3, 3)         # (..., 3, 3)
    goal_obs = obs[..., 25:28]                               # (..., 3)
    vocal_obs = obs[..., 28:31]                              # (..., 3)
    gest_obs = obs[..., 31:34]                               # (..., 3)

    delta_st = st_coords - own_pos.unsqueeze(-2)             # (..., 3, 2)
    at_st = (delta_st.abs().sum(dim=-1) < 0.05).float()      # (..., 3)
    delta_co = 0.5 - own_pos                                 # (..., 2)
    at_co = (delta_co.abs().sum(dim=-1, keepdim=True) < 0.05).float()  # (..., 1)
    delta_partner = partner_pos - own_pos                    # (..., 2)

    # Accessible incoming symbol (delivered via vocal/TTS or visual/caption channel)
    msg_in = (vocal_obs + gest_obs).clamp(0.0, 1.0)          # (..., 3)
    code_cue = goal_obs + msg_in                             # (..., 3)
    # Per-station visual match (strictly 0 when vis_codes == 0, e.g., BLV agent)
    vis_match = (vis_codes * code_cue.unsqueeze(-2)).sum(dim=-1)  # (..., 3)

    base_ego = torch.cat(
        [
            obs[..., :42],                       # 42
            delta_st.reshape(*lead, 6),          # 6
            at_st,                               # 3
            delta_co,                            # 2
            at_co,                               # 1
            delta_partner,                       # 2
            vis_match,                           # 3
        ],
        dim=-1,
    )  # 59 dims
    if obs.shape[-1] > 42:
        return torch.cat([base_ego, obs[..., 42:]], dim=-1)
    return base_ego


class InterdependentCTDENet(nn.Module):
    """Parameter-shared decentralized actor (with role indicator) + centralized critic."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 96,
        use_tom: bool = True,
        use_grounding: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_tom = use_tom
        self.use_grounding = use_grounding

        extra_obs_dim = max(0, obs_dim - 42)
        self.extra_obs_dim = extra_obs_dim
        ego_dim = 59 + extra_obs_dim
        in_dim = ego_dim + NUM_REQ_ACTIONS + NUM_SUPP_ACTIONS
        self.obs_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Online Theory-of-Mind head: infers partner ability profile phi_{-i} & target belief
        self.tom_phi_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, ABILITY_DIM),
            nn.Sigmoid(),
        )
        self.tom_belief_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, NUM_STATIONS),
        )

        # Cross-modal grounding projection (MATE-inspired)
        self.attn_head = nn.Linear(hidden_dim, NUM_STATIONS)
        self.ground_proj = nn.Linear(hidden_dim, 32)

        # Policy receives recurrent state + egocentric station/checkout/extra geometry + ToM
        geom_dim = 6 + 3 + 2 + 1 + 4 + extra_obs_dim
        policy_in_dim = hidden_dim + geom_dim + (ABILITY_DIM + NUM_STATIONS + NUM_STATIONS if use_tom else 0)
        self.policy_trunk = nn.Sequential(
            nn.Linear(policy_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.req_head = nn.Linear(hidden_dim, NUM_REQ_ACTIONS)
        self.supp_head = nn.Linear(hidden_dim, NUM_SUPP_ACTIONS)

        # Centralized Critic V(s) for CTDE training
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def init_hidden(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros((batch_size, 2, self.hidden_dim), dtype=torch.float32, device=device)

    def _heads_from_hidden(
        self, h_flat: torch.Tensor, ego_flat: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        pred_phi = self.tom_phi_head(h_flat)
        pred_belief_logits = self.tom_belief_head(h_flat)
        pred_belief_probs = F.softmax(pred_belief_logits, dim=-1)

        attn_probs = F.softmax(self.attn_head(h_flat), dim=-1)
        ground_emb = F.normalize(self.ground_proj(h_flat), dim=-1)

        # Direct skip of egocentric geometry + own_phi (+ extra domain features if any)
        own_phi = ego_flat[:, 4:8]
        parts = [ego_flat[:, 42:54], own_phi]
        if self.extra_obs_dim > 0:
            parts.append(ego_flat[:, 59 : 59 + self.extra_obs_dim])
        geom = torch.cat(parts, dim=-1)

        if self.use_tom:
            pol_in = torch.cat(
                [h_flat, geom, pred_phi.detach(), pred_belief_probs.detach(), attn_probs.detach()],
                dim=-1,
            )
        else:
            pol_in = torch.cat([h_flat, geom], dim=-1)

        pol_feat = self.policy_trunk(pol_in)
        req_logits = self.req_head(pol_feat)
        supp_logits = self.supp_head(pol_feat)
        return req_logits, supp_logits, pred_phi, pred_belief_logits, attn_probs, ground_emb

    def forward_actor(
        self,
        obs: torch.Tensor,
        prev_req: torch.Tensor,
        prev_supp: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        b = obs.shape[0]
        ego = extract_egocentric_features(obs)
        prev_req_oh = F.one_hot(prev_req, num_classes=NUM_REQ_ACTIONS).float()
        prev_supp_oh = F.one_hot(prev_supp, num_classes=NUM_SUPP_ACTIONS).float()

        ego_flat = ego.reshape(b * 2, -1)
        x = torch.cat([ego, prev_req_oh, prev_supp_oh], dim=-1).reshape(b * 2, -1)
        h_flat = h.reshape(b * 2, self.hidden_dim)

        enc = self.obs_encoder(x)
        h_next_flat = self.gru(enc, h_flat)
        (
            req_logits,
            supp_logits,
            pred_phi,
            pred_belief_logits,
            attn_probs,
            ground_emb,
        ) = self._heads_from_hidden(h_next_flat, ego_flat)

        return (
            req_logits.reshape(b, 2, NUM_REQ_ACTIONS),
            supp_logits.reshape(b, 2, NUM_SUPP_ACTIONS),
            h_next_flat.reshape(b, 2, self.hidden_dim),
            pred_phi.reshape(b, 2, ABILITY_DIM),
            pred_belief_logits.reshape(b, 2, NUM_STATIONS),
            attn_probs.reshape(b, 2, NUM_STATIONS),
            ground_emb.reshape(b, 2, 32),
        )

    def act(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        prev_req: torch.Tensor,
        prev_supp: torch.Tensor,
        h: torch.Tensor,
        deterministic: bool = False,
    ) -> AgentStepOutput:
        (
            req_logits,
            supp_logits,
            h_next,
            pred_phi,
            pred_belief_logits,
            attn_probs,
            ground_emb,
        ) = self.forward_actor(obs, prev_req, prev_supp, h)

        a_req, logp_req, ent_req = _sample_and_logp(req_logits, deterministic=deterministic)
        a_supp, logp_supp, ent_supp = _sample_and_logp(supp_logits, deterministic=deterministic)

        val = self.critic(state).squeeze(-1)
        return AgentStepOutput(
            a_req=a_req,
            a_supp=a_supp,
            logp_req=logp_req,
            logp_supp=logp_supp,
            entropy_joint=ent_req + ent_supp,
            value=val,
            h_next=h_next,
            pred_partner_phi=pred_phi,
            pred_target_logits=pred_belief_logits,
            attn_probs=attn_probs,
            ground_emb=ground_emb,
        )

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        state_seq: torch.Tensor,
        prev_req_seq: torch.Tensor,
        prev_supp_seq: torch.Tensor,
        h_init: torch.Tensor,
        reset_mask_seq: torch.Tensor,
        act_req_seq: torch.Tensor,
        act_supp_seq: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        t_len, mb = obs_seq.shape[:2]
        ego_seq = extract_egocentric_features(obs_seq)
        prev_req_oh = F.one_hot(prev_req_seq, num_classes=NUM_REQ_ACTIONS).float()
        prev_supp_oh = F.one_hot(prev_supp_seq, num_classes=NUM_SUPP_ACTIONS).float()

        ego_flat_all = ego_seq.reshape(t_len * mb * 2, -1)
        x_flat = torch.cat([ego_seq, prev_req_oh, prev_supp_oh], dim=-1).reshape(
            t_len * mb * 2, -1
        )
        enc_seq = self.obs_encoder(x_flat).reshape(t_len, mb * 2, self.hidden_dim)

        h_curr = h_init.reshape(mb * 2, self.hidden_dim)
        h_list = []
        for t in range(t_len):
            rst = reset_mask_seq[t]
            if rst.any():
                rst_pair = rst.unsqueeze(-1).expand(mb, 2).reshape(mb * 2, 1)
                h_curr = torch.where(rst_pair, torch.zeros_like(h_curr), h_curr)
            h_curr = self.gru(enc_seq[t], h_curr)
            h_list.append(h_curr)

        h_all = torch.stack(h_list, dim=0).reshape(t_len * mb * 2, self.hidden_dim)
        (
            req_logits_f,
            supp_logits_f,
            pred_phi_f,
            pred_belief_logits_f,
            attn_probs_f,
            ground_emb_f,
        ) = self._heads_from_hidden(h_all, ego_flat_all)

        req_logits = req_logits_f.reshape(t_len, mb, 2, NUM_REQ_ACTIONS)
        supp_logits = supp_logits_f.reshape(t_len, mb, 2, NUM_SUPP_ACTIONS)

        logp_req, ent_req = _eval_logp_and_ent(req_logits, act_req_seq)
        logp_supp, ent_supp = _eval_logp_and_ent(supp_logits, act_supp_seq)

        val = self.critic(state_seq.reshape(t_len * mb, -1)).reshape(t_len, mb)

        return (
            logp_req,
            logp_supp,
            ent_req + ent_supp,
            val,
            pred_phi_f.reshape(t_len, mb, 2, ABILITY_DIM),
            pred_belief_logits_f.reshape(t_len, mb, 2, NUM_STATIONS),
            attn_probs_f.reshape(t_len, mb, 2, NUM_STATIONS),
            ground_emb_f.reshape(t_len, mb, 2, 32),
            supp_logits,
        )
