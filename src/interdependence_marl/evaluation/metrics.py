"""Interdependence evaluation metrics beyond task return (Xiao et al., Sec. 4.3.3 & 5.3.3).

Measures:
  - Task Completion Rate (`success_rate`)
  - Contribution Symmetry Index (`csi` in [0, 1])
  - Invisible Work / Supportive Effort (`borrower_supp_actions`, `total_supp_actions`)
  - Borrower Agency vs. Paternalistic Override (`consensual_rate`, `override_rate`)
  - Online Theory-of-Mind Capability Estimation Error (`tom_phi_mae`)
  - Assistive Facilitator Intervention Rate (`facilitator_rate`)
"""

from __future__ import annotations

import torch

from interdependence_marl.envs.adc_posg import AbilityDiversePOSG, EnvConfig
from interdependence_marl.models.facilitator import ProToMFacilitator
from interdependence_marl.models.networks import InterdependentCTDENet
from interdependence_marl.profiles import (
    EVAL_DYADS_COMBINING,
    EVAL_DYADS_SHARING,
    make_dyad_batch,
)


@torch.no_grad()
def evaluate_dyad(
    model: InterdependentCTDENet,
    env_cfg: EnvConfig,
    key_0: str,
    key_1: str,
    num_episodes: int = 256,
    facilitator_mode: str = "none",
    deterministic: bool = False,
    seed: int = 1000,
    env_cls: type = AbilityDiversePOSG,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Evaluate a trained model on `num_episodes` parallel rollouts for a specific ability dyad."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    model.eval()

    env = env_cls(num_envs=num_episodes, config=env_cfg, device=dev)
    phi = make_dyad_batch(key_0, key_1, batch_size=num_episodes, device=dev)
    obs, state = env.reset(phi=phi)

    h = model.init_hidden(num_episodes, device=dev)
    prev_req = torch.zeros((num_episodes, 2), dtype=torch.long, device=dev)
    prev_supp = torch.zeros((num_episodes, 2), dtype=torch.long, device=dev)
    facilitator = ProToMFacilitator(mode=facilitator_mode)

    active = torch.ones(num_episodes, dtype=torch.bool, device=dev)
    ep_success = torch.zeros(num_episodes, dtype=torch.bool, device=dev)
    ep_consensual = torch.zeros(num_episodes, dtype=torch.bool, device=dev)
    ep_override = torch.zeros(num_episodes, dtype=torch.bool, device=dev)
    ep_b_supp = torch.zeros(num_episodes, dtype=torch.float32, device=dev)
    ep_tot_supp = torch.zeros(num_episodes, dtype=torch.float32, device=dev)
    ep_csi = torch.zeros(num_episodes, dtype=torch.float32, device=dev)

    phi_mae_accum = 0.0
    phi_mae_steps = 0
    fac_fires = 0.0
    fac_opportunities = 0.0

    for _ in range(env_cfg.max_steps):
        if not active.any():
            break

        step_out = model.act(
            obs, state, prev_req, prev_supp, h, deterministic=deterministic
        )

        # Measure online partner ability inference MAE on active episodes
        true_partner_phi = env.phi.flip(dims=[1])
        mae = (step_out.pred_partner_phi - true_partner_phi).abs().mean(dim=(1, 2))
        phi_mae_accum += float(mae[active].sum().item())
        phi_mae_steps += int(active.sum().item())

        if facilitator_mode != "none":
            relay, fired = facilitator.compute_relay(
                step_out.a_supp, step_out.pred_target_logits, obs
            )
            env.set_relay_messages(relay)
            fac_fires += float(fired[active].mean(dim=-1).sum().item())
            fac_opportunities += float(active.sum().item())

        obs, state, _, done, info = env.step(step_out.a_req, step_out.a_supp)
        h = step_out.h_next
        prev_req = step_out.a_req
        prev_supp = step_out.a_supp

        newly_done = active & done
        if newly_done.any():
            ep_success[newly_done] = info["success"][newly_done]
            ep_consensual[newly_done] = info["consensual_checkout"][newly_done]
            ep_override[newly_done] = info["unilateral_override"][newly_done]
            ep_b_supp[newly_done] = info["borrower_supp_count"][newly_done]
            ep_tot_supp[newly_done] = info["total_supp_count"][newly_done]
            ep_csi[newly_done] = info["csi"][newly_done]
            active[newly_done] = False

    n_succ = int(ep_success.sum().item())
    override_among_succ = (
        float(ep_override[ep_success].float().mean().item()) if n_succ > 0 else 0.0
    )
    consensual_among_succ = (
        float(ep_consensual[ep_success].float().mean().item()) if n_succ > 0 else 0.0
    )

    return {
        "success_rate": float(ep_success.float().mean().item()),
        "consensual_rate": consensual_among_succ,
        "override_rate": override_among_succ,
        "borrower_supp_actions": float(ep_b_supp.mean().item()),
        "total_supp_actions": float(ep_tot_supp.mean().item()),
        "csi": float(ep_csi.mean().item()),
        "tom_phi_mae": phi_mae_accum / max(1, phi_mae_steps),
        "facilitator_rate": fac_fires / max(1.0, fac_opportunities),
    }


def evaluate_suite(
    model: InterdependentCTDENet,
    env_cfg: EnvConfig,
    facilitator_mode: str = "none",
    num_episodes: int = 256,
    env_cls: type = AbilityDiversePOSG,
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, float]]:
    """Evaluate across all canonical seen & unseen ability dyads for the configured mode."""
    dyads = (
        EVAL_DYADS_COMBINING if env_cfg.mode == "ability_combining" else EVAL_DYADS_SHARING
    )
    results: dict[str, dict[str, float]] = {}
    for idx, (label, k0, k1) in enumerate(dyads):
        results[label] = evaluate_dyad(
            model=model,
            env_cfg=env_cfg,
            key_0=k0,
            key_1=k1,
            num_episodes=num_episodes,
            facilitator_mode=facilitator_mode,
            seed=2000 + idx * 17,
            env_cls=env_cls,
            device=device,
        )
    return results
