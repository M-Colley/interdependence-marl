"""Vectorized Recurrent CTDE MAPPO with Online ToM, Cross-Modal Grounding & Ability-UPD."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from interdependence_marl.envs.adc_posg import AbilityDiversePOSG, EnvConfig
from interdependence_marl.models.facilitator import ProToMFacilitator
from interdependence_marl.models.networks import InterdependentCTDENet
from interdependence_marl.training.upd import AbilityUPD


@dataclass
class TrainConfig:
    num_envs: int = 160
    rollout_steps: int = 26
    num_updates: int = 160
    ppo_epochs: int = 4
    num_minibatches: int = 2
    lr: float = 1.8e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.015
    vf_coef: float = 0.5
    tom_coef: float = 0.20
    ground_coef: float = 0.12
    epistemic_pot_scale: float = 0.25
    max_grad_norm: float = 1.0
    hidden_dim: int = 96
    curriculum: str = "ability_upd"  # "homogeneous", "uniform", "ability_upd"
    use_tom: bool = True
    use_grounding: bool = True
    facilitator_train_prob: float = 0.25
    seed: int = 42


def _jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p.log() - m.log())).sum(dim=-1) + (q * (q.log() - m.log())).sum(dim=-1))


def _belief_prob_on_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum of Agent 0 and Agent 1 belief probabilities assigned to `target` `(B,)`."""
    probs = F.softmax(logits, dim=-1)  # (B, 2, 3)
    idx = target.view(-1, 1, 1).expand(-1, 2, 1)
    return probs.gather(-1, idx).squeeze(-1).sum(dim=-1)


def _clipped_ppo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    valid: torch.Tensor,
    clip_coef: float,
) -> torch.Tensor:
    ratio = (new_logp - old_logp).exp()
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * adv
    return (-torch.min(surr1, surr2) * valid).sum() / valid.sum().clamp(min=1.0)


def train_interdependent_mappo(
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
    device: torch.device | str = "cpu",
) -> tuple[InterdependentCTDENet, list[dict[str, float]]]:
    """Train an InterdependentCTDENet policy and return `(model, training_history)`."""
    torch.manual_seed(train_cfg.seed)
    dev = torch.device(device)

    env = AbilityDiversePOSG(num_envs=train_cfg.num_envs, config=env_cfg, device=dev)
    upd = AbilityUPD(
        strategy=train_cfg.curriculum,
        mode=env_cfg.mode,
        device=dev,
    )
    model = InterdependentCTDENet(
        hidden_dim=train_cfg.hidden_dim,
        use_tom=train_cfg.use_tom,
        use_grounding=train_cfg.use_grounding,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, eps=1e-5)
    facilitator = ProToMFacilitator(mode="protom")

    phi_batch, mix_batch, arch_idx = upd.sample_batch(train_cfg.num_envs)
    obs, state = env.reset(phi=phi_batch)
    h = model.init_hidden(train_cfg.num_envs, device=dev)
    prev_req = torch.zeros((train_cfg.num_envs, 2), dtype=torch.long, device=dev)
    prev_supp = torch.zeros((train_cfg.num_envs, 2), dtype=torch.long, device=dev)

    b = train_cfg.num_envs
    t_len = train_cfg.rollout_steps
    history: list[dict[str, float]] = []

    for update in range(1, train_cfg.num_updates + 1):
        progress = (update - 1) / max(1, train_cfg.num_updates - 1)
        optimizer.param_groups[0]["lr"] = train_cfg.lr * (1.0 - 0.75 * progress)

        # Warm up supportive communication cost after agents master basic extraction & grounding
        env.supp_cost_scale = min(1.0, max(0.0, (progress - 0.35) / 0.45))

        use_fac = train_cfg.use_tom and (
            torch.rand(1, device=dev).item() < train_cfg.facilitator_train_prob
        )
        mixed_partner_id = update % 2

        buf_obs = torch.zeros((t_len, b, 2, obs.shape[-1]), device=dev)
        buf_state = torch.zeros((t_len, b, state.shape[-1]), device=dev)
        buf_prev_req = torch.zeros((t_len, b, 2), dtype=torch.long, device=dev)
        buf_prev_supp = torch.zeros((t_len, b, 2), dtype=torch.long, device=dev)
        buf_h_init = h.detach().clone()
        buf_reset_mask = torch.zeros((t_len, b), dtype=torch.bool, device=dev)

        buf_act_req = torch.zeros((t_len, b, 2), dtype=torch.long, device=dev)
        buf_act_supp = torch.zeros((t_len, b, 2), dtype=torch.long, device=dev)
        buf_logp_req = torch.zeros((t_len, b, 2), device=dev)
        buf_logp_supp = torch.zeros((t_len, b, 2), device=dev)
        buf_valid_pg = torch.ones((t_len, b, 2), device=dev)
        buf_val = torch.zeros((t_len, b), device=dev)
        buf_rew = torch.zeros((t_len, b), device=dev)
        buf_done = torch.zeros((t_len, b), dtype=torch.bool, device=dev)

        buf_true_phi = torch.zeros((t_len, b, 2, 4), device=dev)
        buf_comm_target = torch.zeros((t_len, b), dtype=torch.long, device=dev)
        buf_target_station = torch.zeros((t_len, b), dtype=torch.long, device=dev)
        buf_bel_pot = torch.zeros((t_len + 1, b), device=dev)

        ep_successes: list[float] = []
        ep_overrides: list[float] = []
        ep_supp_borrower: list[float] = []

        for t in range(t_len):
            buf_obs[t] = obs
            buf_state[t] = state
            buf_prev_req[t] = prev_req
            buf_prev_supp[t] = prev_supp
            buf_true_phi[t] = env.phi
            buf_target_station[t] = env.target_station
            comm_tgt_now = (
                env.target_code if env_cfg.mode == "ability_sharing" else env.target_station
            )
            buf_comm_target[t] = comm_tgt_now

            with torch.no_grad():
                step_out = model.act(obs, state, prev_req, prev_supp, h)
                if train_cfg.use_tom and train_cfg.use_grounding:
                    buf_bel_pot[t] = _belief_prob_on_target(
                        step_out.pred_target_logits, comm_tgt_now
                    )

            a_req = step_out.a_req.clone()
            a_supp = step_out.a_supp.clone()

            if train_cfg.curriculum == "ability_upd":
                do_mix = torch.rand(b, device=dev) < mix_batch
                if do_mix.any():
                    n_mix = int(do_mix.sum().item())
                    a_req[do_mix, mixed_partner_id] = torch.randint(0, 5, (n_mix,), device=dev)
                    buf_valid_pg[t, do_mix, mixed_partner_id] = 0.0

            if use_fac:
                relay, _ = facilitator.compute_relay(a_supp, step_out.pred_target_logits, obs)
                env.set_relay_messages(relay)

            next_obs, next_state, rew, done, info = env.step(a_req, a_supp)

            buf_act_req[t] = a_req
            buf_act_supp[t] = a_supp
            buf_logp_req[t] = step_out.logp_req
            buf_logp_supp[t] = step_out.logp_supp
            buf_val[t] = step_out.value
            buf_rew[t] = rew
            buf_done[t] = done

            h = step_out.h_next
            prev_req = a_req
            prev_supp = a_supp

            if done.any():
                fin = done
                ep_successes.extend(info["success"][fin].float().tolist())
                if env_cfg.mode == "ability_sharing" and info["success"][fin].any():
                    succ_fin = fin & info["success"]
                    ep_overrides.extend(info["unilateral_override"][succ_fin].float().tolist())
                ep_supp_borrower.extend(info["borrower_supp_count"][fin].tolist())

                new_phi, new_mix, _ = upd.sample_batch(b)
                next_obs, next_state = env.reset(phi=new_phi, env_mask=fin)
                mix_batch = torch.where(fin, new_mix, mix_batch)
                h = torch.where(fin.view(-1, 1, 1), torch.zeros_like(h), h)
                prev_req = torch.where(fin.unsqueeze(-1), torch.zeros_like(prev_req), prev_req)
                prev_supp = torch.where(fin.unsqueeze(-1), torch.zeros_like(prev_supp), prev_supp)
                if t + 1 < t_len:
                    buf_reset_mask[t + 1] = fin

            obs, state = next_obs, next_state

        with torch.no_grad():
            final_out = model.act(obs, state, prev_req, prev_supp, h)
            next_val = final_out.value
            if train_cfg.use_tom and train_cfg.use_grounding:
                comm_tgt_end = (
                    env.target_code if env_cfg.mode == "ability_sharing" else env.target_station
                )
                buf_bel_pot[t_len] = _belief_prob_on_target(
                    final_out.pred_target_logits, comm_tgt_end
                )
                nonterm = (~buf_done).float()
                epistemic_shaping = (
                    train_cfg.epistemic_pot_scale
                    * nonterm
                    * (train_cfg.gamma * buf_bel_pot[1:] - buf_bel_pot[:-1])
                )
                buf_rew = buf_rew + epistemic_shaping

        adv = torch.zeros((t_len, b), device=dev)
        last_gae = torch.zeros(b, device=dev)
        for t in reversed(range(t_len)):
            nonterminal = (~buf_done[t]).float()
            v_next = next_val if t == t_len - 1 else buf_val[t + 1]
            delta = buf_rew[t] + train_cfg.gamma * v_next * nonterminal - buf_val[t]
            last_gae = delta + train_cfg.gamma * train_cfg.gae_lambda * nonterminal * last_gae
            adv[t] = last_gae

        returns = adv + buf_val

        learnability = adv.var(dim=0).detach()
        upd.update_scores(phi_batch, mix_batch, arch_idx, learnability)
        phi_batch = env.phi.clone()

        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        mb_size = b // train_cfg.num_minibatches
        total_tom_loss = 0.0
        total_ground_loss = 0.0

        for _ in range(train_cfg.ppo_epochs):
            perm = torch.randperm(b, device=dev)
            for start in range(0, b, mb_size):
                mb_idx = perm[start : start + mb_size]

                (
                    logp_req_seq,
                    logp_supp_seq,
                    ent_seq,
                    val_seq,
                    pred_phi_seq,
                    pred_bel_seq,
                    attn_seq,
                    ground_seq,
                    supp_logits_seq,
                ) = model.evaluate_sequence(
                    buf_obs[:, mb_idx],
                    buf_state[:, mb_idx],
                    buf_prev_req[:, mb_idx],
                    buf_prev_supp[:, mb_idx],
                    buf_h_init[mb_idx],
                    buf_reset_mask[:, mb_idx],
                    buf_act_req[:, mb_idx],
                    buf_act_supp[:, mb_idx],
                )

                a_mb = adv_norm[:, mb_idx].unsqueeze(-1)
                valid = buf_valid_pg[:, mb_idx]
                pg_req = _clipped_ppo_loss(
                    logp_req_seq, buf_logp_req[:, mb_idx], a_mb, valid, train_cfg.clip_coef
                )
                pg_supp = _clipped_ppo_loss(
                    logp_supp_seq, buf_logp_supp[:, mb_idx], a_mb, valid, train_cfg.clip_coef
                )
                pg_loss = pg_req + pg_supp

                vf_loss = 0.5 * (val_seq - returns[:, mb_idx]).pow(2).mean()
                ent_loss = ent_seq.mean()

                loss = pg_loss + train_cfg.vf_coef * vf_loss - train_cfg.ent_coef * ent_loss

                if train_cfg.use_tom:
                    true_partner_phi = buf_true_phi[:, mb_idx].flip(dims=[2])
                    phi_mse = F.mse_loss(pred_phi_seq, true_partner_phi)
                    c_tgt = buf_comm_target[:, mb_idx].reshape(-1)
                    bel_ce = 0.5 * (
                        F.cross_entropy(pred_bel_seq[:, :, 0].reshape(-1, 3), c_tgt)
                        + F.cross_entropy(pred_bel_seq[:, :, 1].reshape(-1, 3), c_tgt)
                    )
                    t_loss = phi_mse + 0.6 * bel_ce
                    loss = loss + train_cfg.tom_coef * t_loss
                    total_tom_loss = float(t_loss.item())

                if train_cfg.use_grounding:
                    # Grounded object-attention & referential binding (MATE-inspired, collapse-free):
                    # grounds each agent's station attention to task-salient station `st_tgt` (with
                    # symmetric stop-grad distillation) and binds symbols {0,1,2} to inferred belief.
                    st_tgt = buf_target_station[:, mb_idx].reshape(-1)
                    st_sup = 0.5 * (
                        F.nll_loss((attn_seq[:, :, 0].reshape(-1, 3) + 1e-7).log(), st_tgt)
                        + F.nll_loss((attn_seq[:, :, 1].reshape(-1, 3) + 1e-7).log(), st_tgt)
                    )
                    # Stop-grad teacher-student alignment from higher-confidence agent to partner
                    a0, a1 = attn_seq[:, :, 0], attn_seq[:, :, 1]
                    c0 = a0.max(dim=-1, keepdim=True).values
                    c1 = a1.max(dim=-1, keepdim=True).values
                    teacher = torch.where(c0 >= c1, a0.detach(), a1.detach()).clamp(min=1e-7)
                    student = torch.where(c0 >= c1, a1, a0).clamp(min=1e-7)
                    asym_kl = (teacher * (teacher.log() - student.log())).sum(dim=-1).mean()

                    bel_probs = F.softmax(pred_bel_seq.detach(), dim=-1)
                    bel_conf = bel_probs.max(dim=-1).values
                    vocal_logp = F.log_softmax(supp_logits_seq[:, :, :, 1:4], dim=-1)
                    gest_logp = F.log_softmax(supp_logits_seq[:, :, :, 4:7], dim=-1)
                    ref_kl = -(bel_probs * (vocal_logp + gest_logp) * 0.5).sum(dim=-1)
                    ref_loss = (ref_kl * (bel_conf > 0.65).float()).mean()

                    g_loss = st_sup + 0.25 * asym_kl + 0.5 * ref_loss
                    loss = loss + train_cfg.ground_coef * g_loss
                    total_ground_loss = float(g_loss.item())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                optimizer.step()

        mean_succ = sum(ep_successes) / max(1, len(ep_successes))
        mean_over = sum(ep_overrides) / max(1, len(ep_overrides)) if ep_overrides else 0.0
        mean_supp = sum(ep_supp_borrower) / max(1, len(ep_supp_borrower))
        history.append(
            {
                "update": float(update),
                "env_steps": float(update * b * t_len),
                "train_success": float(mean_succ),
                "train_override": float(mean_over),
                "train_borrower_supp": float(mean_supp),
                "tom_loss": total_tom_loss,
                "ground_loss": total_ground_loss,
            }
        )

    return model, history
