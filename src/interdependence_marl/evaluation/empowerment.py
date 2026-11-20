"""Information-Theoretic Empowerment & Counterfactual Shapley Contribution Metrics.

Provides domain-agnostic agency and interdependence measures:
  1. Exact Counterfactual Discrete Empowerment `I(A_0; S_{t+1} | s_t)` (in nats)
  2. Barber-Agakov Variational Empowerment `VariationalEmpowermentNet` for continuous control
  3. Exact 2-Player Counterfactual Shapley Values `(phi_0^Shap, phi_1^Shap)` & Synergy Surplus
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn

from interdependence_marl.envs.adc_posg import NUM_REQ_ACTIONS, AbilityDiversePOSG, EnvConfig
from interdependence_marl.models.networks import InterdependentCTDENet
from interdependence_marl.profiles import make_dyad_batch


@torch.no_grad()
def compute_discrete_empowerment(
    model: InterdependentCTDENet,
    env_cfg: EnvConfig,
    key_0: str,
    key_1: str,
    num_episodes: int = 128,
    mc_samples_per_action: int = 4,
    seed: int = 3000,
    device: torch.device | str = "cpu",
) -> float:
    """Compute Borrower (Agent 0) counterfactual action-state mutual information `I(A_0; S' | s)` in nats.

    At mid-episode interaction states, branches the environment across all 6 `a_req` actions
    of Agent 0 (while Partner 1 executes `pi_1`) and computes exact discrete KL divergence
    between `P(S' | s, a_0)` and marginal `P(S' | s)` over task-relevant state signatures
    `(pos_0_x, pos_0_y, held_by, checkout_success)`.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    model.eval()

    env = AbilityDiversePOSG(num_envs=num_episodes, config=env_cfg, device=dev)
    phi = make_dyad_batch(key_0, key_1, batch_size=num_episodes, device=dev)
    obs, state = env.reset(phi=phi)
    h = model.init_hidden(num_episodes, device=dev)
    prev_req = torch.zeros((num_episodes, 2), dtype=torch.long, device=dev)
    prev_supp = torch.zeros((num_episodes, 2), dtype=torch.long, device=dev)

    mi_accum = 0.0
    mi_count = 0
    active = torch.ones(num_episodes, dtype=torch.bool, device=dev)

    for step in range(min(12, env_cfg.max_steps)):
        if not active.any():
            break

        step_out = model.act(obs, state, prev_req, prev_supp, h)

        # Probe counterfactual empowerment on active states near interaction milestones
        if step in (2, 5, 7):
            act_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            if act_idx.numel() > 0:
                # Encode discrete outcome signature in [0, 5*5*4*2 = 200)
                n_act = act_idx.numel()
                cond_counts = torch.zeros(
                    (n_act, NUM_REQ_ACTIONS, 200), dtype=torch.float32, device=dev
                )
                for a0_cand in range(NUM_REQ_ACTIONS):
                    for _ in range(mc_samples_per_action):
                        sim_env = copy.deepcopy(env)
                        cand_req = step_out.a_req.clone()
                        cand_req[:, 0] = a0_cand
                        _, _, _, _, sim_info = sim_env.step(cand_req, step_out.a_supp)
                        p0 = sim_env.pos[act_idx, 0]
                        hb = (sim_env.held_by[act_idx] + 1).clamp(0, 3)
                        sc = sim_info["success"][act_idx].long()
                        sig = ((p0[:, 0] * 5 + p0[:, 1]) * 4 + hb) * 2 + sc
                        cond_counts[
                            torch.arange(n_act, device=dev), a0_cand, sig
                        ] += 1.0 / mc_samples_per_action

                # P(S' | s, a_0) has shape (N_act, 6, 200); marginal P(S' | s) is mean over a_0
                p_cond = cond_counts.clamp(min=1e-9)
                p_marg = cond_counts.mean(dim=1, keepdim=True).clamp(min=1e-9)
                kl = (cond_counts * (p_cond.log() - p_marg.log())).sum(dim=-1).mean(dim=-1)
                mi_accum += float(kl.sum().item())
                mi_count += int(n_act)

        obs, state, _, done, _ = env.step(step_out.a_req, step_out.a_supp)
        h, prev_req, prev_supp = step_out.h_next, step_out.a_req, step_out.a_supp
        active = active & (~done)

    return mi_accum / max(1, mi_count)


@torch.no_grad()
def compute_shapley_values(
    model: InterdependentCTDENet,
    env_cfg: EnvConfig,
    key_0: str,
    key_1: str,
    num_episodes: int = 256,
    seed: int = 4000,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Compute exact 2-player counterfactual Shapley values and supermodular synergy surplus."""
    dev = torch.device(device)
    model.eval()

    coalitions = {
        "empty": (False, False),
        "only_0": (True, False),
        "only_1": (False, True),
        "grand": (True, True),
    }
    vals: dict[str, float] = {}

    for name, (on_0, on_1) in coalitions.items():
        torch.manual_seed(seed)
        env = AbilityDiversePOSG(num_envs=num_episodes, config=env_cfg, device=dev)
        phi = make_dyad_batch(key_0, key_1, batch_size=num_episodes, device=dev)
        obs, state = env.reset(phi=phi)
        h = model.init_hidden(num_episodes, device=dev)
        prev_req = torch.zeros((num_episodes, 2), dtype=torch.long, device=dev)
        prev_supp = torch.zeros((num_episodes, 2), dtype=torch.long, device=dev)

        active = torch.ones(num_episodes, dtype=torch.bool, device=dev)
        ep_succ = torch.zeros(num_episodes, dtype=torch.bool, device=dev)

        for _ in range(env_cfg.max_steps):
            if not active.any():
                break
            out = model.act(obs, state, prev_req, prev_supp, h)
            a_req = out.a_req.clone()
            a_supp = out.a_supp.clone()
            if not on_0:
                a_req[:, 0] = 0
                a_supp[:, 0] = 0
            if not on_1:
                a_req[:, 1] = 0
                a_supp[:, 1] = 0

            obs, state, _, done, info = env.step(a_req, a_supp)
            h, prev_req, prev_supp = out.h_next, a_req, a_supp
            newly_done = active & done
            ep_succ[newly_done] = info["success"][newly_done]
            active = active & (~done)

        vals[name] = float(ep_succ.float().mean().item())

    shap_0 = 0.5 * (vals["only_0"] - vals["empty"]) + 0.5 * (vals["grand"] - vals["only_1"])
    shap_1 = 0.5 * (vals["only_1"] - vals["empty"]) + 0.5 * (vals["grand"] - vals["only_0"])
    synergy = vals["grand"] - vals["only_0"] - vals["only_1"] + vals["empty"]

    return {
        "v_empty": vals["empty"],
        "v_only_0": vals["only_0"],
        "v_only_1": vals["only_1"],
        "v_grand": vals["grand"],
        "shapley_0": shap_0,
        "shapley_1": shap_1,
        "synergy_surplus": synergy,
    }


class VariationalEmpowermentNet(nn.Module):
    """Barber-Agakov Gaussian inverse-dynamics estimator `q_xi(a_user | s_t, s_{t+k})` for continuous control.

    Estimates lower bound on `I(A_user; S_{t+k} | S_t) = H(A_user | S_t) - H(A_user | S_t, S_{t+k})`.
    """

    def __init__(self, state_dim: int, action_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.logstd_head = nn.Linear(hidden_dim, action_dim)

    def log_prob(
        self, s_now: torch.Tensor, s_future: torch.Tensor, a_user: torch.Tensor
    ) -> torch.Tensor:
        feat = self.net(torch.cat([s_now, s_future], dim=-1))
        mean = torch.tanh(self.mean_head(feat))
        logstd = self.logstd_head(feat).clamp(-3.0, 1.0)
        var = (2.0 * logstd).exp()
        logp = -0.5 * (((a_user - mean).pow(2) / var) + 2.0 * logstd + math.log(2.0 * math.pi))
        return logp.sum(dim=-1)

    def estimate_mi(
        self,
        s_now: torch.Tensor,
        s_future: torch.Tensor,
        a_user: torch.Tensor,
        prior_std: float = 0.55,
    ) -> torch.Tensor:
        """Return per-sample variational mutual information estimate in nats."""
        # Differential entropy of prior N(0, prior_std^2 I_d) + E[log q_xi(a | s, s')]
        d = a_user.shape[-1]
        h_prior = d * (0.5 * math.log(2.0 * math.pi * math.e * (prior_std ** 2)))
        log_q = self.log_prob(s_now, s_future, a_user)
        return (h_prior + log_q).clamp(min=0.0)
