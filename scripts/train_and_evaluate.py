"""End-to-end training, ablation study, and figure generation for Interdependence-MARL."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interdependence_marl.envs.adc_posg import EnvConfig  # noqa: E402
from interdependence_marl.evaluation.metrics import evaluate_dyad, evaluate_suite  # noqa: E402
from interdependence_marl.training.mappo import TrainConfig, train_interdependent_mappo  # noqa: E402


def smooth(vals: list[float], weight: float = 0.80) -> list[float]:
    if not vals:
        return []
    out = [vals[0]]
    for v in vals[1:]:
        out.append(weight * out[-1] + (1.0 - weight) * v)
    return out


def plot_combining_results(
    histories: dict[str, list[dict[str, float]]],
    eval_suites: dict[str, dict[str, dict[str, float]]],
    cross_modal_fac: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1.45, 1.0]})

    colors = {
        "Homogeneous-SP": "#8c969e",
        "Uniform-DR (No ToM)": "#e68a2e",
        "ADC-MARL (Ours)": "#1b6ca8",
    }

    # Panel A: Training Success Curve
    ax = axes[0]
    for name, hist in histories.items():
        steps = [h["env_steps"] / 1e3 for h in hist]
        succ = smooth([h["train_success"] * 100.0 for h in hist], weight=0.82)
        ax.plot(steps, succ, label=name, color=colors[name], linewidth=2.2)
    ax.set_title("A. Training Convergence (Ability Combining)", fontweight="bold")
    ax.set_xlabel("Environment Steps (x10^3)")
    ax.set_ylabel("Training Episode Success (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="lower right")

    # Panel B: Zero-Shot Evaluation Across Seen & Unseen Dyads (excluding orthogonal cross-modal)
    ax = axes[1]
    dyad_labels = [
        "Symmetric Full-Ability",
        "Wheelchair + BLV (Vending)",
        "Wheelchair-DHH + Full",
        "Motor-Ataxia + BLV",
        "LowVision-Tremor + Wheelchair",
    ]
    short_labels = [
        "Full +\nFull",
        "Wheelchair +\nBLV",
        "Wheelchair-DHH\n+ Full",
        "Motor-Ataxia\n+ BLV",
        "LowVis-Tremor\n+ Wheelchair",
    ]
    x = np.arange(len(dyad_labels))
    width = 0.26
    for idx, name in enumerate(["Homogeneous-SP", "Uniform-DR (No ToM)", "ADC-MARL (Ours)"]):
        vals = [eval_suites[name][d]["success_rate"] * 100.0 for d in dyad_labels]
        ax.bar(
            x + (idx - 1) * width,
            vals,
            width=width,
            label=name,
            color=colors[name],
            edgecolor="white",
        )
    ax.set_title("B. Zero-Shot Cross-Ability Generalization", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8.8)
    ax.set_ylabel("Task Completion Rate (%)")
    ax.set_ylim(0, 122)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8.2)

    # Panel C: ProToM Facilitator on Orthogonal Sensory Dyad (BLV + Wheelchair-DHH)
    ax = axes[2]
    fac_names = ["No Facilitator", "Always-On Relay", "ProToM (Ours)"]
    fac_keys = ["none", "always_on", "protom"]
    succ_vals = [cross_modal_fac[k]["success_rate"] * 100.0 for k in fac_keys]
    fire_vals = [cross_modal_fac[k]["facilitator_rate"] * 100.0 for k in fac_keys]

    fx = np.arange(len(fac_names))
    w2 = 0.34
    ax.bar(fx - w2 / 2, succ_vals, width=w2, color="#1b6ca8", label="Task Success (%)")
    ax.bar(fx + w2 / 2, fire_vals, width=w2, color="#d9534f", alpha=0.85, label="Alert/Relay Load (%)")
    ax.set_title("C. Cross-Modal Dyad (BLV + DHH)", fontweight="bold")
    ax.set_xticks(fx)
    ax.set_xticklabels(fac_names, fontsize=8.8)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper left", fontsize=8.8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sharing_results(
    sharing_suites: dict[str, dict[str, dict[str, float]]],
    sharing_protom: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))

    dyads = [
        "BLV Borrower + Sighted Provider",
        "Wheelchair Borrower + Ambulatory Provider",
        "LowVision-Tremor Borrower + Sighted Provider",
        "Wheelchair Borrower + BLV Provider (PwD->PwD)",
    ]
    short = [
        "BLV ->\nSighted",
        "Wheelchair ->\nAmbulatory",
        "LowVis-Tremor\n-> Sighted",
        "Wheelchair ->\nBLV (PwD-PwD)",
    ]
    x = np.arange(len(dyads))
    w = 0.34

    # Panel A: Task Completion Rate
    ax = axes[0]
    v_task = [sharing_suites["Task-Only MARL"][d]["success_rate"] * 100.0 for d in dyads]
    v_adc = [sharing_suites["ADC-MARL (Ours)"][d]["success_rate"] * 100.0 for d in dyads]
    ax.bar(x - w / 2, v_task, width=w, color="#e68a2e", label="Task-Only MARL")
    ax.bar(x + w / 2, v_adc, width=w, color="#1b6ca8", label="ADC-MARL (Interdependence)")
    ax.set_title("A. Ability Sharing: Task Completion", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 122)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    # Panel B: Paternalistic Unilateral Override Rate (lower is better!)
    ax = axes[1]
    o_task = [sharing_suites["Task-Only MARL"][d]["override_rate"] * 100.0 for d in dyads]
    o_adc = [sharing_suites["ADC-MARL (Ours)"][d]["override_rate"] * 100.0 for d in dyads]
    ax.bar(x - w / 2, o_task, width=w, color="#d9534f", label="Task-Only MARL")
    ax.bar(x + w / 2, o_adc, width=w, color="#2a9d8f", label="ADC-MARL (Interdependence)")
    ax.set_title("B. Paternalistic Override Rate (Lower = Better)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel("Unilateral Checkout Override (%)")
    ax.set_ylim(0, 122)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    # Panel C: Borrower Invisible Work (Mean Supportive Utterances per Episode)
    ax = axes[2]
    iw_task = [sharing_suites["Task-Only MARL"][d]["borrower_supp_actions"] for d in dyads]
    iw_adc = [sharing_suites["ADC-MARL (Ours)"][d]["borrower_supp_actions"] for d in dyads]
    ax.bar(x - w / 2, iw_task, width=w, color="#e68a2e", label="Task-Only MARL")
    ax.bar(x + w / 2, iw_adc, width=w, color="#1b6ca8", label="ADC-MARL (Interdependence)")
    ax.set_title("C. Borrower 'Invisible Work' (Lower = Better)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel("Borrower Supportive Actions / Ep")
    ax.set_ylim(0, 15.5)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(4)
    t0 = time.time()
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    res_dir = ROOT / "results"
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: Training Ability-Combining MARL Ablations ===")
    comb_env_cfg = EnvConfig(mode="ability_combining", interdependence_reward=True)
    comb_specs = {
        "Homogeneous-SP": TrainConfig(
            num_envs=160,
            num_updates=160,
            hidden_dim=96,
            curriculum="homogeneous",
            use_tom=False,
            use_grounding=False,
            facilitator_train_prob=0.0,
            seed=11,
        ),
        "Uniform-DR (No ToM)": TrainConfig(
            num_envs=160,
            num_updates=160,
            hidden_dim=96,
            curriculum="uniform",
            use_tom=False,
            use_grounding=False,
            facilitator_train_prob=0.0,
            seed=22,
        ),
        "ADC-MARL (Ours)": TrainConfig(
            num_envs=160,
            num_updates=170,
            hidden_dim=96,
            curriculum="ability_upd",
            use_tom=True,
            use_grounding=True,
            facilitator_train_prob=0.25,
            seed=33,
        ),
    }

    comb_histories: dict[str, list[dict[str, float]]] = {}
    comb_eval: dict[str, dict[str, dict[str, float]]] = {}
    comb_adc_model = None

    for name, tcfg in comb_specs.items():
        t_start = time.time()
        model, hist = train_interdependent_mappo(comb_env_cfg, tcfg)
        comb_histories[name] = hist
        suite = evaluate_suite(model, comb_env_cfg, facilitator_mode="none", num_episodes=256)
        comb_eval[name] = suite
        if name == "ADC-MARL (Ours)":
            comb_adc_model = model
            torch.save(model.state_dict(), ckpt_dir / "adc_marl_combining.pt")
        vending_succ = suite["Wheelchair + BLV (Vending)"]["success_rate"] * 100.0
        print(
            f"  [{name:20s}] done in {time.time() - t_start:5.1f}s | "
            f"Wheelchair+BLV Zero-Shot Success: {vending_succ:5.1f}%"
        )

    assert comb_adc_model is not None
    cross_modal_combining = {}
    for fac_mode in ("none", "always_on", "protom"):
        cross_modal_combining[fac_mode] = evaluate_dyad(
            comb_adc_model,
            comb_env_cfg,
            "blv_ambulatory",
            "wheelchair_dhh",
            num_episodes=256,
            facilitator_mode=fac_mode,
        )
    print(
        "  [ProToM Combining Cross-Modal (BLV + Wheelchair-DHH)] "
        f"none={cross_modal_combining['none']['success_rate']*100:.1f}% | "
        f"always_on={cross_modal_combining['always_on']['success_rate']*100:.1f}% "
        f"(load={cross_modal_combining['always_on']['facilitator_rate']*100:.1f}%) | "
        f"protom={cross_modal_combining['protom']['success_rate']*100:.1f}% "
        f"(load={cross_modal_combining['protom']['facilitator_rate']*100:.1f}%)"
    )

    print("\n=== Phase 2: Training Ability-Sharing MARL (Task-Only vs. Interdependence) ===")
    sharing_suites: dict[str, dict[str, dict[str, float]]] = {}
    sharing_models = {}

    for label, use_interdep, seed in [
        ("Task-Only MARL", False, 44),
        ("ADC-MARL (Ours)", True, 55),
    ]:
        t_start = time.time()
        scfg = EnvConfig(mode="ability_sharing", interdependence_reward=use_interdep)
        tcfg = TrainConfig(
            num_envs=160,
            num_updates=170,
            hidden_dim=96,
            curriculum="ability_upd",
            use_tom=True,
            use_grounding=True,
            facilitator_train_prob=0.25,
            seed=seed,
        )
        model, _ = train_interdependent_mappo(scfg, tcfg)
        sharing_models[label] = model
        eval_cfg = EnvConfig(mode="ability_sharing", interdependence_reward=True)
        suite = evaluate_suite(model, eval_cfg, facilitator_mode="none", num_episodes=256)
        sharing_suites[label] = suite
        if use_interdep:
            torch.save(model.state_dict(), ckpt_dir / "adc_marl_sharing.pt")

        blv_res = suite["BLV Borrower + Sighted Provider"]
        print(
            f"  [{label:20s}] done in {time.time() - t_start:5.1f}s | "
            f"BLV->Sighted Succ: {blv_res['success_rate']*100:5.1f}% | "
            f"Override: {blv_res['override_rate']*100:5.1f}% | "
            f"Invisible Work: {blv_res['borrower_supp_actions']:4.2f} acts/ep"
        )

    sharing_protom = {}
    eval_scfg = EnvConfig(mode="ability_sharing", interdependence_reward=True)
    for fac_mode in ("none", "always_on", "protom"):
        sharing_protom[fac_mode] = evaluate_dyad(
            sharing_models["ADC-MARL (Ours)"],
            eval_scfg,
            "blv_ambulatory",
            "dhh_sighted",
            num_episodes=256,
            facilitator_mode=fac_mode,
        )
    print(
        "  [ProToM Sharing Cross-Modal (BLV Borrower + DHH Provider)] "
        f"none={sharing_protom['none']['success_rate']*100:.1f}% | "
        f"always_on={sharing_protom['always_on']['success_rate']*100:.1f}% "
        f"(load={sharing_protom['always_on']['facilitator_rate']*100:.1f}%) | "
        f"protom={sharing_protom['protom']['success_rate']*100:.1f}% "
        f"(load={sharing_protom['protom']['facilitator_rate']*100:.1f}%)"
    )

    plot_combining_results(
        comb_histories,
        comb_eval,
        cross_modal_combining,
        fig_dir / "ability_combining_results.png",
    )
    plot_sharing_results(
        sharing_suites,
        sharing_protom,
        fig_dir / "ability_sharing_agency.png",
    )

    summary = {
        "ability_combining": comb_eval,
        "ability_combining_cross_modal_facilitator": cross_modal_combining,
        "ability_sharing": sharing_suites,
        "ability_sharing_cross_modal_facilitator": sharing_protom,
        "wall_time_sec": round(time.time() - t0, 2),
    }
    with open(res_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll experiments finished in {time.time() - t0:.1f}s. Results saved to {res_dir}.")


if __name__ == "__main__":
    main()
