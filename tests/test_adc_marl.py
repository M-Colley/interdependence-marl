"""Unit tests verifying formal properties of Tiers 1-3, Empowerment, Shapley, Primal-Dual & stats."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interdependence_marl.envs.adc_posg import (  # noqa: E402
    OBS_DIM,
    STATE_DIM,
    AbilityDiversePOSG,
    EnvConfig,
)
from interdependence_marl.envs.continuous_assistive import (  # noqa: E402
    ContinuousBiomechanicalEnv,
    train_and_eval_tier3_biomechanical,
)
from interdependence_marl.envs.overcooked_ability import OvercookedAbilityEnv  # noqa: E402
from interdependence_marl.evaluation.empowerment import (  # noqa: E402
    VariationalEmpowermentNet,
    compute_discrete_empowerment,
    compute_shapley_values,
)
from interdependence_marl.evaluation.rliable_stats import (  # noqa: E402
    probability_of_improvement,
    stratified_bootstrap_iqm,
)
from interdependence_marl.models.facilitator import ProToMFacilitator  # noqa: E402
from interdependence_marl.models.networks import InterdependentCTDENet  # noqa: E402
from interdependence_marl.profiles import make_dyad_batch  # noqa: E402
from interdependence_marl.training.lagrangian import PrimalDualController  # noqa: E402
from interdependence_marl.training.mappo import TrainConfig, train_interdependent_mappo  # noqa: E402


class TestInterdependenceMARL(unittest.TestCase):
    def test_sensory_masking_no_leakage(self) -> None:
        """Verify BLV agents get strictly zeroed visual features and DHH get zeroed vocal signals."""
        cfg = EnvConfig(mode="ability_combining")
        env = AbilityDiversePOSG(num_envs=32, config=cfg)
        phi = make_dyad_batch("blv_ambulatory", "dhh_sighted", batch_size=32)
        obs, state = env.reset(phi=phi)

        self.assertEqual(obs.shape, (32, 2, OBS_DIM))
        self.assertEqual(state.shape, (32, STATE_DIM))
        self.assertTrue(torch.all(obs[:, 0, 16:28] == 0.0))
        self.assertTrue(torch.all(obs[:, 1, 25:28].sum(dim=-1) == 1.0))

        a_req = torch.zeros((32, 2), dtype=torch.long)
        a_supp = torch.tensor([[1, 4]], dtype=torch.long).expand(32, 2)
        next_obs, _, _, _, _ = env.step(a_req, a_supp)
        self.assertTrue(torch.all(next_obs[:, 1, 28:31] == 0.0))
        self.assertTrue(torch.all(next_obs[:, 0, 31:34] == 0.0))

    def test_strict_structural_interdependence_wheelchair_solo(self) -> None:
        """Verify a wheelchair user (reach=0) cannot extract from high dispensers solo."""
        cfg = EnvConfig(mode="ability_combining")
        env = AbilityDiversePOSG(num_envs=16, config=cfg)
        phi = make_dyad_batch("wheelchair_sighted", "blv_ambulatory", batch_size=16)
        env.reset(phi=phi)

        b_idx = torch.arange(16)
        env.pos[:, 0] = env.station_pos[b_idx, env.target_station]
        env.pos[:, 1] = env.checkout_pos

        a_req = torch.tensor([[5, 0]], dtype=torch.long).expand(16, 2)
        a_supp = torch.zeros((16, 2), dtype=torch.long)
        env.step(a_req, a_supp)
        self.assertTrue(torch.all(env.held_by == -1))
        self.assertTrue(torch.all(env.last_reach_fail[:, 0] == 1.0))

    def test_ability_sharing_private_goal_and_agency(self) -> None:
        """Verify Provider never directly observes Borrower's private goal in Ability Sharing."""
        cfg = EnvConfig(mode="ability_sharing", interdependence_reward=True)
        env = AbilityDiversePOSG(num_envs=16, config=cfg)
        phi = make_dyad_batch("blv_ambulatory", "full_ability", batch_size=16)
        obs, _ = env.reset(phi=phi)

        self.assertTrue(torch.all(obs[:, 0, 25:28].sum(dim=-1) == 1.0))
        self.assertTrue(torch.all(obs[:, 1, 25:28] == 0.0))

    def test_protom_selective_gating(self) -> None:
        """Verify ProToM intervenes when beliefs diverge across modality gap and quiets once aligned."""
        fac = ProToMFacilitator(mode="protom", jsd_threshold=0.08)
        cfg = EnvConfig(mode="ability_sharing")
        env = AbilityDiversePOSG(num_envs=8, config=cfg)
        phi = make_dyad_batch("blv_ambulatory", "dhh_sighted", batch_size=8)
        obs, _ = env.reset(phi=phi)

        a_supp = torch.tensor([[1, 0]], dtype=torch.long).expand(8, 2)
        div_logits = torch.tensor([[[5.0, -2.0, -2.0], [0.0, 0.0, 0.0]]]).expand(8, 2, 3)
        relay, fired = fac.compute_relay(a_supp, div_logits, obs)
        self.assertTrue(torch.all(fired[:, 1] == 1.0))
        self.assertTrue(torch.all(relay[:, 1, 0] == 1.0))

        aligned_logits = torch.tensor([[[5.0, -3.0, -3.0], [5.0, -3.0, -3.0]]]).expand(8, 2, 3)
        _, fired_aligned = fac.compute_relay(torch.zeros_like(a_supp), aligned_logits, obs)
        self.assertTrue(torch.all(fired_aligned == 0.0))

    def test_primal_dual_lagrangian_and_fcp_snapshots(self) -> None:
        """Verify PrimalDualController dual ascent and FCP snapshot co-play."""
        ctrl = PrimalDualController()
        nu0 = ctrl.nu_override
        out = ctrl.step_duals(progress=0.5, batch_override_rate=0.60, batch_burden=9.0, batch_empowerment=0.1)
        self.assertGreater(out["nu_override"], nu0)

        snaps: list[InterdependentCTDENet] = []
        env_cfg = EnvConfig(mode="ability_sharing", interdependence_reward=True)
        t_cfg = TrainConfig(
            num_envs=16, rollout_steps=12, num_updates=3, hidden_dim=48, use_lagrangian=True
        )
        model, hist = train_interdependent_mappo(env_cfg, t_cfg, snapshots_out=snaps)
        self.assertIsInstance(model, InterdependentCTDENet)
        self.assertGreaterEqual(len(snaps), 2)
        self.assertIn("nu_override", hist[-1])

        # Train FCP agent against snapshot pool
        fcp_model, _ = train_interdependent_mappo(env_cfg, t_cfg, partner_pool=snaps)
        self.assertIsInstance(fcp_model, InterdependentCTDENet)

    def test_empowerment_shapley_and_rliable(self) -> None:
        """Verify discrete empowerment, Shapley value identities, and rliable bootstrap CIs."""
        cfg = EnvConfig(mode="ability_combining")
        net = InterdependentCTDENet(hidden_dim=48)
        mi = compute_discrete_empowerment(net, cfg, "wheelchair_sighted", "blv_ambulatory", num_episodes=16)
        self.assertGreaterEqual(mi, 0.0)

        shap = compute_shapley_values(net, cfg, "wheelchair_sighted", "blv_ambulatory", num_episodes=32)
        # Efficiency axiom of Shapley values: shap_0 + shap_1 == v(grand) - v(empty)
        self.assertAlmostEqual(
            shap["shapley_0"] + shap["shapley_1"],
            shap["v_grand"] - shap["v_empty"],
            places=5,
        )

        mat_a = np.array([[0.90, 0.95], [0.92, 0.96], [0.89, 0.94]])
        mat_b = np.array([[0.30, 0.40], [0.28, 0.45], [0.32, 0.38]])
        stats = stratified_bootstrap_iqm(mat_a, num_bootstraps=200)
        self.assertLessEqual(stats["ci_low"], stats["iqm"])
        self.assertGreaterEqual(stats["ci_high"], stats["iqm"])
        self.assertEqual(probability_of_improvement(mat_a, mat_b), 1.0)

    def test_tier2_overcooked_and_tier3_biomechanical(self) -> None:
        """Verify Tier 2 OvercookedAbilityEnv and Tier 3 ContinuousBiomechanicalEnv."""
        oc_cfg = EnvConfig(max_steps=20, mode="ability_combining")
        oc_env = OvercookedAbilityEnv(num_envs=16, config=oc_cfg)
        obs, state = oc_env.reset(phi=make_dyad_batch("wheelchair_sighted", "blv_ambulatory", 16))
        self.assertEqual(obs.shape, (16, 2, 46))
        self.assertEqual(state.shape, (16, 54))

        m_oc, _ = train_interdependent_mappo(
            oc_cfg,
            TrainConfig(num_envs=16, rollout_steps=14, num_updates=2, hidden_dim=48),
            env_cls=OvercookedAbilityEnv,
        )
        self.assertIsInstance(m_oc, InterdependentCTDENet)

        bio_env = ContinuousBiomechanicalEnv(num_envs=16)
        b_obs = bio_env.reset()
        self.assertEqual(b_obs.shape, (16, 9))
        v_emp = VariationalEmpowermentNet(state_dim=4, action_dim=2)
        mi_est = v_emp.estimate_mi(torch.randn(16, 4), torch.randn(16, 4), torch.randn(16, 2))
        self.assertEqual(mi_est.shape, (16,))

        res_t3 = train_and_eval_tier3_biomechanical(num_updates=3, num_envs=32)
        self.assertIn("ADC-Lagrangian Assist (Ours)", res_t3["metrics"])


if __name__ == "__main__":
    unittest.main()
