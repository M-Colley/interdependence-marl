"""Unit tests verifying formal properties of the Ability-Diverse Collaboration POSG & MARL stack."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interdependence_marl.envs.adc_posg import (  # noqa: E402
    OBS_DIM,
    STATE_DIM,
    AbilityDiversePOSG,
    EnvConfig,
)
from interdependence_marl.models.facilitator import ProToMFacilitator  # noqa: E402
from interdependence_marl.models.networks import InterdependentCTDENet  # noqa: E402
from interdependence_marl.profiles import make_dyad_batch  # noqa: E402
from interdependence_marl.training.mappo import TrainConfig, train_interdependent_mappo  # noqa: E402
from interdependence_marl.training.upd import AbilityUPD  # noqa: E402


class TestInterdependenceMARL(unittest.TestCase):
    def test_sensory_masking_no_leakage(self) -> None:
        """Verify BLV agents get strictly zeroed visual features and DHH get zeroed vocal signals."""
        cfg = EnvConfig(mode="ability_combining")
        env = AbilityDiversePOSG(num_envs=32, config=cfg)
        # Agent 0: BLV (vis=0, aud=1); Agent 1: DHH (vis=1, aud=0)
        phi = make_dyad_batch("blv_ambulatory", "dhh_sighted", batch_size=32)
        obs, state = env.reset(phi=phi)

        self.assertEqual(obs.shape, (32, 2, OBS_DIM))
        self.assertEqual(state.shape, (32, STATE_DIM))

        # Agent 0 (BLV) must have strictly zero station visual codes [16:25] & visual goal [25:28]
        self.assertTrue(torch.all(obs[:, 0, 16:28] == 0.0))
        # Agent 1 (DHH-Sighted) perceives visual goal cleanly
        self.assertTrue(torch.all(obs[:, 1, 25:28].sum(dim=-1) == 1.0))

        # Step with Agent 0 emitting vocal utterance (1) and Agent 1 emitting visual gesture (4)
        a_req = torch.zeros((32, 2), dtype=torch.long)
        a_supp = torch.tensor([[1, 4]], dtype=torch.long).expand(32, 2)
        next_obs, _, _, _, _ = env.step(a_req, a_supp)

        # Agent 1 is DHH (aud=0) -> cannot hear Agent 0's vocal signal [28:31]
        self.assertTrue(torch.all(next_obs[:, 1, 28:31] == 0.0))
        # Agent 0 is BLV (vis=0) -> cannot see Agent 1's visual gesture [31:34]
        self.assertTrue(torch.all(next_obs[:, 0, 31:34] == 0.0))

    def test_strict_structural_interdependence_wheelchair_Solo(self) -> None:
        """Verify a wheelchair user (reach=0) cannot extract from high dispensers solo."""
        cfg = EnvConfig(mode="ability_combining")
        env = AbilityDiversePOSG(num_envs=16, config=cfg)
        phi = make_dyad_batch("wheelchair_sighted", "blv_ambulatory", batch_size=16)
        env.reset(phi=phi)

        # Place Agent 0 (wheelchair, reach=0) directly onto target station & try to interact alone
        b_idx = torch.arange(16)
        env.pos[:, 0] = env.station_pos[b_idx, env.target_station]
        env.pos[:, 1] = env.checkout_pos  # Partner far away at center (dist=4 > 1)

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

        # Borrower (0) sees target_code; Provider (1) sees strictly zeros in goal slot [25:28]
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
        # Divergent beliefs: Agent 0 confident in class 0, Agent 1 uniform
        div_logits = torch.tensor([[[5.0, -2.0, -2.0], [0.0, 0.0, 0.0]]]).expand(8, 2, 3)
        relay, fired = fac.compute_relay(a_supp, div_logits, obs)
        self.assertTrue(torch.all(fired[:, 1] == 1.0))
        self.assertTrue(torch.all(relay[:, 1, 0] == 1.0))

        # Aligned confident beliefs: JSD == 0 and low entropy -> ProToM stays silent!
        aligned_logits = torch.tensor([[[5.0, -3.0, -3.0], [5.0, -3.0, -3.0]]]).expand(8, 2, 3)
        _, fired_aligned = fac.compute_relay(torch.zeros_like(a_supp), aligned_logits, obs)
        self.assertTrue(torch.all(fired_aligned == 0.0))

    def test_upd_and_short_mappo_run(self) -> None:
        """Smoke test Ability-UPD archive update and end-to-end Recurrent MAPPO training."""
        upd = AbilityUPD(strategy="ability_upd", mode="ability_combining")
        phi, mix, idx = upd.sample_batch(32)
        self.assertEqual(phi.shape, (32, 2, 4))
        upd.update_scores(phi, mix, idx, torch.rand(32))

        env_cfg = EnvConfig(mode="ability_combining")
        t_cfg = TrainConfig(num_envs=16, rollout_steps=14, num_updates=2, hidden_dim=64)
        model, hist = train_interdependent_mappo(env_cfg, t_cfg)
        self.assertIsInstance(model, InterdependentCTDENet)
        self.assertEqual(len(hist), 2)


if __name__ == "__main__":
    unittest.main()
