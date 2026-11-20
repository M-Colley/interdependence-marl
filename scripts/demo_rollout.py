"""Step-by-step interpretable episode trace of trained ADC-MARL policies."""

from __future__ import annotations

from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interdependence_marl.envs.adc_posg import AbilityDiversePOSG, EnvConfig  # noqa: E402
from interdependence_marl.models.networks import InterdependentCTDENet  # noqa: E402
from interdependence_marl.profiles import make_dyad_batch  # noqa: E402

REQ_NAMES = ["STAY", "UP", "DOWN", "LEFT", "RIGHT", "INTERACT"]
SUPP_NAMES = ["silent", "vocal:0", "vocal:1", "vocal:2", "gest:0", "gest:1", "gest:2"]


@torch.no_grad()
def run_trace(ckpt_path: Path, mode: str, key_0: str, key_1: str, title: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"TRACE: {title} ({mode})")
    print(f"{'=' * 78}")

    cfg = EnvConfig(mode=mode, interdependence_reward=True)
    env = AbilityDiversePOSG(num_envs=1, config=cfg)
    model = InterdependentCTDENet(hidden_dim=96, use_tom=True, use_grounding=True)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    torch.manual_seed(7)
    phi = make_dyad_batch(key_0, key_1, batch_size=1)
    obs, state = env.reset(phi=phi)
    h = model.init_hidden(1)
    prev_req = torch.zeros((1, 2), dtype=torch.long)
    prev_supp = torch.zeros((1, 2), dtype=torch.long)

    st_pos = env.station_pos[0].tolist()
    st_codes = env.station_vis_code[0].tolist()
    t_st = int(env.target_station[0].item())
    t_code = int(env.target_code[0].item())
    print(f"Stations (r,c) & vis codes: {list(zip(st_pos, st_codes))}")
    print(f"Target Station #{t_st} at {st_pos[t_st]} (Visual Code = {t_code}) | Checkout = [2, 2]")
    print(f"True phi_0 ({key_0}): {phi[0, 0].tolist()} | True phi_1 ({key_1}): {phi[0, 1].tolist()}\n")

    for step in range(1, cfg.max_steps + 1):
        out = model.act(obs, state, prev_req, prev_supp, h, deterministic=True)
        r0, r1 = int(out.a_req[0, 0]), int(out.a_req[0, 1])
        s0, s1 = int(out.a_supp[0, 0]), int(out.a_supp[0, 1])
        hat_phi1 = [round(x, 2) for x in out.pred_partner_phi[0, 0].tolist()]
        hat_phi0 = [round(x, 2) for x in out.pred_partner_phi[0, 1].tolist()]

        obs, state, rew, done, info = env.step(out.a_req, out.a_supp)
        h, prev_req, prev_supp = out.h_next, out.a_req, out.a_supp

        p0 = env.pos[0, 0].tolist()
        p1 = env.pos[0, 1].tolist()
        held = int(env.held_by[0].item())
        print(
            f"t={step:02d} | A0@{p0} ({REQ_NAMES[r0]:8s}, {SUPP_NAMES[s0]:7s}) hat_phi1={hat_phi1} | "
            f"A1@{p1} ({REQ_NAMES[r1]:8s}, {SUPP_NAMES[s1]:7s}) hat_phi0={hat_phi0} | held={held}"
        )
        if bool(done[0].item()):
            print(
                f"--> Episode Done: success={bool(info['success'][0])} | "
                f"consensual={bool(info['consensual_checkout'][0])} | "
                f"override={bool(info['unilateral_override'][0])} | "
                f"CSI={float(info['csi'][0]):.2f} | rew={float(rew[0]):.2f}"
            )
            break


if __name__ == "__main__":
    ckpt_comb = ROOT / "checkpoints" / "adc_marl_combining.pt"
    ckpt_shar = ROOT / "checkpoints" / "adc_marl_sharing.pt"
    if ckpt_comb.exists():
        run_trace(
            ckpt_comb,
            "ability_combining",
            "wheelchair_sighted",
            "blv_ambulatory",
            "Wheelchair-Sighted + BLV-Ambulatory (Vending Machine Synergy)",
        )
    if ckpt_shar.exists():
        run_trace(
            ckpt_shar,
            "ability_sharing",
            "blv_ambulatory",
            "full_ability",
            "BLV Borrower + Sighted Provider (Guided Shopping with Agency)",
        )
