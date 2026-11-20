"""Master Paper Benchmark Suite: Multi-Seed `rliable` CIs, FCP, Ablations, Primal-Dual & Tiers 2-3."""

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
from interdependence_marl.envs.continuous_assistive import (  # noqa: E402
    train_and_eval_tier3_biomechanical,
)
from interdependence_marl.envs.overcooked_ability import OvercookedAbilityEnv  # noqa: E402
from interdependence_marl.evaluation.empowerment import (  # noqa: E402
    compute_discrete_empowerment,
    compute_shapley_values,
)
from interdependence_marl.evaluation.metrics import evaluate_suite  # noqa: E402
from interdependence_marl.evaluation.rliable_stats import (  # noqa: E402
    probability_of_improvement,
    stratified_bootstrap_iqm,
)
from interdependence_marl.models.networks import InterdependentCTDENet  # noqa: E402
from interdependence_marl.training.mappo import TrainConfig, train_interdependent_mappo  # noqa: E402

ASYM_COMB_DYADS = [
    "Wheelchair + BLV (Vending)",
    "Wheelchair-DHH + Full",
    "Motor-Ataxia + BLV",
    "LowVision-Tremor + Wheelchair",
]


def plot_tier1_ablations_and_lagrangian(
    iqm_summary: dict[str, dict[str, float]],
    shapley_res: dict[str, dict[str, float]],
    lag_hist: list[dict[str, float]],
    sharing_comp: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    plt.rcParams.update({"font.size": 9.8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.7), gridspec_kw={"width_ratios": [1.3, 1.1, 1.15]})

    # Panel A: Stratified Bootstrap IQM + 95% CIs across Asymmetric Dyads (3 seeds)
    ax = axes[0]
    methods = list(iqm_summary.keys())
    iqms = [iqm_summary[m]["iqm"] * 100.0 for m in methods]
    err_low = [iqms[i] - iqm_summary[m]["ci_low"] * 100.0 for i, m in enumerate(methods)]
    err_high = [iqm_summary[m]["ci_high"] * 100.0 - iqms[i] for i, m in enumerate(methods)]
    palette = ["#8c969e", "#c97b63", "#e68a2e", "#5aa9e6", "#4ea8de", "#1b6ca8"]

    y = np.arange(len(methods))
    ax.barh(
        y,
        iqms,
        xerr=[err_low, err_high],
        color=palette[: len(methods)],
        capsize=4,
        height=0.62,
        edgecolor="white",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=8.8)
    ax.set_xlabel("Zero-Shot Asymmetric Success IQM (%) [95% Bootstrap CI]")
    ax.set_xlim(0, 105)
    ax.set_title("A. Tier 1 Ablations & FCP (rliable IQM)", fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    # Panel B: Primal-Dual Multiplier & Override Dynamics in CA-POSG
    ax = axes[1]
    steps = [h["env_steps"] / 1e3 for h in lag_hist]
    ema_ov = [h.get("ema_override", 0.0) * 100.0 for h in lag_hist]
    nu_ov = [h.get("nu_override", 0.0) * 100.0 for h in lag_hist]
    ax.plot(steps, ema_ov, color="#d9534f", linewidth=2.1, label="Override Rate EMA (%)")
    ax.axhline(5.0, color="#d9534f", linestyle=":", linewidth=1.3, label="Constraint tau=5%")
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(steps, nu_ov, color="#2a9d8f", linewidth=2.0, linestyle="--", label="Dual nu_override (x100)")
    ax.set_xlabel("Environment Steps (x10^3)")
    ax.set_ylabel("Unilateral Override (%)", color="#d9534f")
    ax2.set_ylabel("Dual Multiplier (x100)", color="#2a9d8f")
    ax.set_ylim(0, 100)
    ax.set_title("B. Primal-Dual CA-POSG Saddle Trajectory", fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right", fontsize=8.2)
    ax.grid(alpha=0.22, linestyle="--")

    # Panel C: Information-Theoretic Empowerment & Counterfactual Shapley Synergy
    ax = axes[2]
    cats = ["Task-Only\nSharing", "Lagrangian\nADC-MARL", "Homo-SP\n(Vending)", "ADC-MARL\n(Vending)"]
    emp_vals = [
        sharing_comp["Task-Only"]["empowerment_nats"],
        sharing_comp["Primal-Dual ADC-MARL"]["empowerment_nats"],
        shapley_res["Homogeneous-SP"]["synergy_surplus"],
        shapley_res["ADC-MARL (Full)"]["synergy_surplus"],
    ]
    bar_cols = ["#e68a2e", "#1b6ca8", "#8c969e", "#2a9d8f"]
    bx = np.arange(len(cats))
    ax.bar(bx, emp_vals, width=0.52, color=bar_cols, edgecolor="white")
    ax.set_xticks(bx)
    ax.set_xticklabels(cats, fontsize=8.5)
    ax.set_ylabel("Empowerment (nats)  |  Shapley Synergy")
    ax.set_title("C. Borrower Empowerment & Shapley Synergy", fontweight="bold")
    ax.axvline(1.5, color="#cbd5e1", linestyle="--")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tier2_overcooked(
    oc_suites: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10.5, 4.5))

    dyads = [
        "Symmetric Full-Ability",
        "Wheelchair + BLV (Vending)",
        "Wheelchair-DHH + Full",
        "Motor-Ataxia + BLV",
        "LowVision-Tremor + Wheelchair",
    ]
    short = [
        "Full + Full\nChefs",
        "Wheelchair +\nBLV Chefs",
        "Wheelchair-DHH\n+ Full",
        "Motor-Ataxia\n+ BLV",
        "LowVis-Tremor\n+ Wheelchair",
    ]
    methods = ["Homogeneous-SP", "FCP (Co-Play)", "Uniform-DR", "ADC-MARL (Ours)"]
    cols = ["#8c969e", "#c97b63", "#e68a2e", "#1b6ca8"]
    x = np.arange(len(dyads))
    w = 0.19

    for i, m in enumerate(methods):
        vals = [oc_suites[m][d]["success_rate"] * 100.0 for d in dyads]
        ax.bar(x + (i - 1.5) * w, vals, width=w, label=m, color=cols[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("3-Stage Soup Order Completion (%)")
    ax.set_ylim(0, 118)
    ax.set_title(
        "Tier 2 (`OvercookedV2-Ability`): Procedural Pantry -> Cooking Pot -> Serving Window",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8.8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tier3_biomechanical(t3_data: dict[str, object], out_path: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    metrics: dict[str, dict[str, float]] = t3_data["metrics"]  # type: ignore[assignment]
    trajs: dict[str, dict[str, list[list[float]]]] = t3_data["trajectories"]  # type: ignore[assignment]

    # Panel A: Sample 2D Limb Trajectories under 5Hz Pathological Tremor + ROM Saturation
    ax = axes[0]
    tx, ty = trajs["target"]["xy"][0]
    circle = plt.Circle((tx, ty), 0.14, color="#dcfce7", ec="#16a34a", lw=1.8, label="Target Zone")
    ax.add_patch(circle)

    style_map = {
        "Unassisted Impaired": ("#8c969e", "--", 1.6),
        "Task-Only Shared Autonomy": ("#d9534f", "-.", 2.0),
        "ADC-Lagrangian Assist (Ours)": ("#1b6ca8", "-", 2.4),
    }
    for name, (col, ls, lw) in style_map.items():
        pts = np.asarray(trajs[name]["xy"])
        ax.plot(pts[:, 0], pts[:, 1], color=col, linestyle=ls, linewidth=lw, label=name)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=col, s=42, zorder=5)

    ax.scatter([0.0], [0.0], color="#1e293b", marker="s", s=45, label="Origin")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("Workspace X")
    ax.set_ylabel("Workspace Y")
    ax.set_title("A. 2D Musculoskeletal Tremor Trajectories", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.3, loc="best")
    ax.grid(alpha=0.25, linestyle="--")

    # Panel B: Task Stabilization vs. Variational User Empowerment I_q(A_u; S_{t+3} | S_t)
    ax = axes[1]
    names = list(style_map.keys())
    short = ["Unassisted\nImpaired", "Task-Only\nTakeover", "ADC-Lagrangian\nAssist (Ours)"]
    reach = [metrics[n]["target_reach_rate"] * 100.0 for n in names]
    emp = [metrics[n]["user_empowerment_nats"] * 100.0 for n in names]  # scaled x100 for dual bar

    x = np.arange(len(names))
    w = 0.34
    ax.bar(x - w / 2, reach, width=w, color="#1b6ca8", label="Target Stabilization (%)")
    ax.bar(x + w / 2, emp, width=w, color="#2a9d8f", label="User Empowerment (x0.01 nats)")
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("Score / Scaled Nats")
    ax.set_ylim(0, 125)
    ax.set_title("B. Tier 3 Task Stabilization vs. User Empowerment", fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=8.8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(4)
    t_global = time.time()
    res_dir = ROOT / "results"
    fig_dir = res_dir / "figures"
    ckpt_dir = ROOT / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    seeds = [101, 202, 303]
    comb_cfg = EnvConfig(mode="ability_combining", interdependence_reward=True)

    print("=== Block 1: Tier 1 Multi-Seed Ablations, FCP & rliable Bootstrap CIs ===")
    fcp_pool: list[InterdependentCTDENet] = []
    method_matrices: dict[str, list[list[float]]] = {
        "Homogeneous-SP": [],
        "FCP (Co-Play)": [],
        "ADC (-UPD / Uniform)": [],
        "ADC (-ToM)": [],
        "ADC (-EpistemicPot)": [],
        "ADC-MARL (Full)": [],
    }
    saved_models: dict[str, InterdependentCTDENet] = {}

    # 1a. Train Homogeneous-SP across 3 seeds & harvest checkpoint pool for FCP
    for s in seeds:
        snaps: list[InterdependentCTDENet] = []
        m, _ = train_interdependent_mappo(
            comb_cfg,
            TrainConfig(
                num_envs=128,
                num_updates=135,
                hidden_dim=80,
                curriculum="homogeneous",
                use_tom=False,
                use_grounding=False,
                use_epistemic_pot=False,
                facilitator_train_prob=0.0,
                seed=s,
            ),
            snapshots_out=snaps if len(fcp_pool) < 6 else None,
        )
        fcp_pool.extend(snaps)
        suite = evaluate_suite(m, comb_cfg, num_episodes=192)
        method_matrices["Homogeneous-SP"].append([suite[d]["success_rate"] for d in ASYM_COMB_DYADS])
        saved_models["Homogeneous-SP"] = m
    print("  [Homogeneous-SP (3 seeds)] complete; harvested", len(fcp_pool), "FCP checkpoints.")

    # 1b. Train FCP (Strouse et al., NeurIPS 2021) against the multi-checkpoint partner pool
    for s in seeds:
        m, _ = train_interdependent_mappo(
            comb_cfg,
            TrainConfig(
                num_envs=128,
                num_updates=135,
                hidden_dim=80,
                curriculum="homogeneous",
                use_tom=False,
                use_grounding=False,
                use_epistemic_pot=False,
                facilitator_train_prob=0.0,
                seed=s + 10,
            ),
            partner_pool=fcp_pool,
        )
        suite = evaluate_suite(m, comb_cfg, num_episodes=192)
        method_matrices["FCP (Co-Play)"].append([suite[d]["success_rate"] for d in ASYM_COMB_DYADS])
    print("  [FCP (Co-Play)  (3 seeds)] complete.")

    # 1c. Train Ablations & Full ADC-MARL across 3 seeds
    ablation_configs = [
        ("ADC (-UPD / Uniform)", "uniform", True, True, True),
        ("ADC (-ToM)", "ability_upd", False, False, False),
        ("ADC (-EpistemicPot)", "ability_upd", True, True, False),
        ("ADC-MARL (Full)", "ability_upd", True, True, True),
    ]
    for label, curr, utom, ugrd, uep in ablation_configs:
        t0 = time.time()
        for s in seeds:
            m, _ = train_interdependent_mappo(
                comb_cfg,
                TrainConfig(
                    num_envs=128,
                    num_updates=145,
                    hidden_dim=80,
                    curriculum=curr,
                    use_tom=utom,
                    use_grounding=ugrd,
                    use_epistemic_pot=uep,
                    facilitator_train_prob=0.25 if utom else 0.0,
                    seed=s + 50,
                ),
            )
            suite = evaluate_suite(m, comb_cfg, num_episodes=192)
            method_matrices[label].append([suite[d]["success_rate"] for d in ASYM_COMB_DYADS])
            saved_models[label] = m
        iqm_now = stratified_bootstrap_iqm(np.asarray(method_matrices[label]))
        print(
            f"  [{label:20s} (3 seeds)] {time.time()-t0:5.1f}s | "
            f"IQM={iqm_now['iqm']*100:5.1f}% [{iqm_now['ci_low']*100:4.1f}, {iqm_now['ci_high']*100:4.1f}]"
        )

    iqm_summary = {
        m: stratified_bootstrap_iqm(np.asarray(mat), seed=idx)
        for idx, (m, mat) in enumerate(method_matrices.items())
    }
    p_imp_vs_fcp = probability_of_improvement(
        np.asarray(method_matrices["ADC-MARL (Full)"]),
        np.asarray(method_matrices["FCP (Co-Play)"]),
    )
    print(f"  P(ADC-MARL > FCP) = {p_imp_vs_fcp:.3f}")

    # Compute Exact Counterfactual Shapley Values on Wheelchair + BLV
    shapley_res = {
        k: compute_shapley_values(
            saved_models[k], comb_cfg, "wheelchair_sighted", "blv_ambulatory", num_episodes=256
        )
        for k in ("Homogeneous-SP", "ADC-MARL (Full)")
    }
    print(
        "  [Shapley Synergy (Wheelchair+BLV)] "
        f"Homo-SP={shapley_res['Homogeneous-SP']['synergy_surplus']:.3f} | "
        f"ADC-MARL={shapley_res['ADC-MARL (Full)']['synergy_surplus']:.3f}"
    )

    print("\n=== Block 2: Primal-Dual Lagrangian CA-POSG & Information-Theoretic Empowerment ===")
    task_only_cfg = EnvConfig(mode="ability_sharing", interdependence_reward=False)
    m_task_only, _ = train_interdependent_mappo(
        task_only_cfg,
        TrainConfig(num_envs=144, num_updates=155, hidden_dim=96, seed=404),
    )
    torch.save(m_task_only.state_dict(), ckpt_dir / "task_only_sharing.pt")

    lag_env_cfg = EnvConfig(mode="ability_sharing", interdependence_reward=True)
    m_lag, lag_hist = train_interdependent_mappo(
        lag_env_cfg,
        TrainConfig(
            num_envs=144,
            num_updates=165,
            hidden_dim=96,
            use_lagrangian=True,
            seed=505,
        ),
    )

    eval_sh_cfg = EnvConfig(mode="ability_sharing", interdependence_reward=True)
    s_to = evaluate_suite(m_task_only, eval_sh_cfg, num_episodes=256)
    s_lg = evaluate_suite(m_lag, eval_sh_cfg, num_episodes=256)

    emp_to = compute_discrete_empowerment(
        m_task_only, eval_sh_cfg, "blv_ambulatory", "full_ability", num_episodes=96
    )
    emp_lg = compute_discrete_empowerment(
        m_lag, eval_sh_cfg, "blv_ambulatory", "full_ability", num_episodes=96
    )

    sharing_comp = {
        "Task-Only": {
            "blv_success": s_to["BLV Borrower + Sighted Provider"]["success_rate"],
            "blv_override": s_to["BLV Borrower + Sighted Provider"]["override_rate"],
            "blv_burden": s_to["BLV Borrower + Sighted Provider"]["borrower_supp_actions"],
            "wheelchair_override": s_to["Wheelchair Borrower + Ambulatory Provider"]["override_rate"],
            "empowerment_nats": emp_to,
        },
        "Primal-Dual ADC-MARL": {
            "blv_success": s_lg["BLV Borrower + Sighted Provider"]["success_rate"],
            "blv_override": s_lg["BLV Borrower + Sighted Provider"]["override_rate"],
            "blv_burden": s_lg["BLV Borrower + Sighted Provider"]["borrower_supp_actions"],
            "wheelchair_override": s_lg["Wheelchair Borrower + Ambulatory Provider"]["override_rate"],
            "empowerment_nats": emp_lg,
            "final_duals": {
                "nu_override": lag_hist[-1].get("nu_override", 0.0),
                "nu_burden": lag_hist[-1].get("nu_burden", 0.0),
                "nu_emp": lag_hist[-1].get("nu_emp", 0.0),
            },
        },
    }
    print(
        "  [Primal-Dual Sharing] "
        f"Succ={sharing_comp['Primal-Dual ADC-MARL']['blv_success']*100:.1f}% | "
        f"Override={sharing_comp['Primal-Dual ADC-MARL']['blv_override']*100:.1f}% "
        f"(vs Task-Only {sharing_comp['Task-Only']['blv_override']*100:.1f}%) | "
        f"Empowerment={emp_lg:.3f} nats (vs {emp_to:.3f})"
    )

    plot_tier1_ablations_and_lagrangian(
        iqm_summary,
        shapley_res,
        lag_hist,
        sharing_comp,
        fig_dir / "tier1_ablations_and_lagrangian.png",
    )

    print("\n=== Block 3: Tier 2 `OvercookedV2-Ability` Procedural Multi-Stage Kitchen ===")
    oc_cfg = EnvConfig(max_steps=32, mode="ability_combining", interdependence_reward=True)
    oc_fcp_pool: list[InterdependentCTDENet] = []
    oc_suites: dict[str, dict[str, dict[str, float]]] = {}

    m_oc_homo, _ = train_interdependent_mappo(
        oc_cfg,
        TrainConfig(
            num_envs=144,
            rollout_steps=32,
            num_updates=160,
            hidden_dim=96,
            curriculum="homogeneous",
            use_tom=False,
            use_grounding=False,
            use_epistemic_pot=False,
            facilitator_train_prob=0.0,
            seed=601,
        ),
        snapshots_out=oc_fcp_pool,
        env_cls=OvercookedAbilityEnv,
    )
    oc_suites["Homogeneous-SP"] = evaluate_suite(
        m_oc_homo, oc_cfg, num_episodes=256, env_cls=OvercookedAbilityEnv
    )

    m_oc_fcp, _ = train_interdependent_mappo(
        oc_cfg,
        TrainConfig(
            num_envs=144,
            rollout_steps=32,
            num_updates=160,
            hidden_dim=96,
            curriculum="homogeneous",
            use_tom=False,
            use_grounding=False,
            use_epistemic_pot=False,
            facilitator_train_prob=0.0,
            seed=602,
        ),
        partner_pool=oc_fcp_pool,
        env_cls=OvercookedAbilityEnv,
    )
    oc_suites["FCP (Co-Play)"] = evaluate_suite(
        m_oc_fcp, oc_cfg, num_episodes=256, env_cls=OvercookedAbilityEnv
    )

    m_oc_udr, _ = train_interdependent_mappo(
        oc_cfg,
        TrainConfig(
            num_envs=144,
            rollout_steps=32,
            num_updates=160,
            hidden_dim=96,
            curriculum="uniform",
            use_tom=False,
            use_grounding=False,
            use_epistemic_pot=False,
            facilitator_train_prob=0.0,
            seed=603,
        ),
        env_cls=OvercookedAbilityEnv,
    )
    oc_suites["Uniform-DR"] = evaluate_suite(
        m_oc_udr, oc_cfg, num_episodes=256, env_cls=OvercookedAbilityEnv
    )

    m_oc_adc, _ = train_interdependent_mappo(
        oc_cfg,
        TrainConfig(
            num_envs=144,
            rollout_steps=32,
            num_updates=175,
            hidden_dim=96,
            curriculum="ability_upd",
            use_tom=True,
            use_grounding=True,
            use_epistemic_pot=True,
            facilitator_train_prob=0.25,
            seed=604,
        ),
        env_cls=OvercookedAbilityEnv,
    )
    oc_suites["ADC-MARL (Ours)"] = evaluate_suite(
        m_oc_adc, oc_cfg, num_episodes=256, env_cls=OvercookedAbilityEnv
    )
    for k in oc_suites:
        v = oc_suites[k]["Wheelchair + BLV (Vending)"]["success_rate"] * 100.0
        print(f"  [OvercookedV2-Ability | {k:18s}] Wheelchair+BLV Chefs: {v:5.1f}%")

    plot_tier2_overcooked(oc_suites, fig_dir / "tier2_overcooked_ability.png")

    print("\n=== Block 4: Tier 3 Continuous Biomechanical Tremor & Variational Empowerment ===")
    t3_res = train_and_eval_tier3_biomechanical(num_updates=95, num_envs=192, seed=707)
    for k, v in t3_res["metrics"].items():  # type: ignore[union-attr]
        print(
            f"  [Tier 3 | {k:28s}] Reach={v['target_reach_rate']*100:5.1f}% | "
            f"Empowerment={v['user_empowerment_nats']:.3f} nats"
        )
    plot_tier3_biomechanical(t3_res, fig_dir / "tier3_biomechanical_empowerment.png")

    full_out = {
        "tier1_rliable_iqm": iqm_summary,
        "tier1_prob_adc_beats_fcp": p_imp_vs_fcp,
        "tier1_shapley_wheelchair_blv": shapley_res,
        "tier1_primal_dual_sharing": sharing_comp,
        "tier2_overcooked_ability": oc_suites,
        "tier3_biomechanical": t3_res["metrics"],
        "wall_time_sec": round(time.time() - t_global, 2),
    }
    with open(res_dir / "enhanced_paper_metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_out, f, indent=2)

    print(f"\nEnhanced 3-Tier paper suite finished in {time.time() - t_global:.1f}s.")


if __name__ == "__main__":
    main()
