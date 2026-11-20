"""Ability-Conditioned Unsupervised Partner Design (Ability-UPD).

Extends Unsupervised Partner Design (Ruhdorfer, Bortoletto, Oei, Penzkofer, & Bulling,
ICML 2026) to the continuous multi-dimensional ability space Phi x Phi = [0, 1]^4 x [0, 1]^4.
Maintains an archive of asymmetric ability dyads and policy-mixing coefficients, prioritizing
configurations with maximal GAE/TD-error variance (the agent's learnability frontier).
"""

from __future__ import annotations

import torch
from torch.distributions import Beta


class AbilityUPD:
    """Population-free curriculum over partner ability vectors and policy-mixing weights."""

    def __init__(
        self,
        strategy: str = "ability_upd",
        mode: str = "ability_combining",
        archive_size: int = 64,
        replay_prob: float = 0.70,
        temperature: float = 0.7,
        device: torch.device | str = "cpu",
    ) -> None:
        assert strategy in ("homogeneous", "uniform", "ability_upd")
        self.strategy = strategy
        self.mode = mode
        self.archive_size = archive_size
        self.replay_prob = replay_prob
        self.temperature = temperature
        self.device = torch.device(device)

        self.beta_dist = Beta(
            torch.tensor(0.45, device=self.device),
            torch.tensor(0.45, device=self.device),
        )
        self.archive_phi = self._generate_viable_candidates(archive_size)
        self.archive_mix = torch.rand(archive_size, device=self.device) * 0.06
        self.archive_scores = torch.ones(archive_size, device=self.device)

    def _generate_viable_candidates(self, n: int) -> torch.Tensor:
        """Sample continuous ability pairs ensuring collective task solvability."""
        if self.strategy == "homogeneous":
            return torch.ones((n, 2, 4), dtype=torch.float32, device=self.device)

        if self.strategy == "uniform":
            phi = torch.rand((n, 2, 4), dtype=torch.float32, device=self.device)
            # Ensure collective reach & vision aren't simultaneously impossible
            phi[:, 1, 0] = torch.maximum(phi[:, 1, 0], 0.5 - phi[:, 0, 0]).clamp(0.0, 1.0)
            phi[:, 1, 2] = torch.maximum(phi[:, 1, 2], 0.6 - phi[:, 0, 2]).clamp(0.0, 1.0)
            return phi

        # Ability-UPD: U-shaped Beta(0.45, 0.45) samples both sharp boundary impairments
        # (near 0.0) and intermediate partial abilities, snapped at <0.08 to exact 0.0
        phi = self.beta_dist.sample((n, 2, 4))
        phi = torch.where(phi < 0.12, torch.zeros_like(phi), phi)
        phi = torch.where(phi > 0.90, torch.ones_like(phi), phi)

        if self.mode == "ability_sharing":
            # Provider (Agent 1) supplies required visual/reach capability while Borrower (Agent 0)
            # spans diverse sensory/motor profiles (and vice-versa for PwD->PwD sharing)
            pwd_to_pwd = torch.rand(n, device=self.device) < 0.25
            phi[:, 1, 2] = torch.where(pwd_to_pwd, torch.ones_like(phi[:, 1, 2]), phi[:, 1, 2].clamp(min=0.75))
            phi[:, 1, 0] = torch.where(pwd_to_pwd, phi[:, 1, 0], phi[:, 1, 0].clamp(min=0.75))
            phi[:, 0, 0] = torch.where(pwd_to_pwd, torch.ones_like(phi[:, 0, 0]), phi[:, 0, 0])
            phi[:, 0, 2] = torch.where(pwd_to_pwd, torch.zeros_like(phi[:, 0, 2]), phi[:, 0, 2])
        else:
            # Ensure collective solvability: at least one agent perceives visual code and at least
            # one agent (or cooperative lift) can actuate the high dispenser
            no_vis = (phi[:, 0, 0] < 0.35) & (phi[:, 1, 0] < 0.35)
            give_0_vis = torch.rand(n, device=self.device) < 0.5
            phi[:, 0, 0] = torch.where(no_vis & give_0_vis, torch.ones_like(phi[:, 0, 0]), phi[:, 0, 0])
            phi[:, 1, 0] = torch.where(no_vis & (~give_0_vis), torch.ones_like(phi[:, 1, 0]), phi[:, 1, 0])

            no_reach = (phi[:, 0, 2] + phi[:, 1, 2]) < 0.75
            give_1_reach = torch.rand(n, device=self.device) < 0.5
            phi[:, 1, 2] = torch.where(no_reach & give_1_reach, torch.ones_like(phi[:, 1, 2]), phi[:, 1, 2])
            phi[:, 0, 2] = torch.where(no_reach & (~give_1_reach), torch.ones_like(phi[:, 0, 2]), phi[:, 0, 2])

        # Guarantee at least one shared sensory channel unless assistive facilitator is active
        both_blocked = ((phi[:, 0, 1] * phi[:, 1, 1]) < 0.2) & ((phi[:, 0, 0] * phi[:, 1, 0]) < 0.2)
        restore_aud = both_blocked & (torch.rand(n, device=self.device) < 0.75)
        phi[:, 0, 1] = torch.where(restore_aud, torch.ones_like(phi[:, 0, 1]), phi[:, 0, 1])
        phi[:, 1, 1] = torch.where(restore_aud, torch.ones_like(phi[:, 1, 1]), phi[:, 1, 1])

        phi[:, :, 3] = phi[:, :, 3].clamp(min=0.40)  # Bounded tremor/slip
        return phi.clamp(0.0, 1.0)

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(phi_batch (B, 2, 4), mix_alpha (B,), archive_indices (B,))`."""
        if self.strategy in ("homogeneous", "uniform"):
            phi = self._generate_viable_candidates(batch_size)
            mix = torch.zeros(batch_size, device=self.device)
            idx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
            return phi, mix, idx

        # Prioritized archive sampling + fresh mutation exploration
        probs = (self.archive_scores + 1e-4).pow(1.0 / self.temperature)
        probs = probs / probs.sum()

        use_archive = torch.rand(batch_size, device=self.device) < self.replay_prob
        arch_idx = torch.multinomial(probs, batch_size, replacement=True)

        fresh_phi = self._generate_viable_candidates(batch_size)
        fresh_mix = torch.rand(batch_size, device=self.device) * 0.06

        phi = torch.where(use_archive.view(-1, 1, 1), self.archive_phi[arch_idx], fresh_phi)
        mix = torch.where(use_archive, self.archive_mix[arch_idx], fresh_mix)
        tracked_idx = torch.where(
            use_archive, arch_idx, torch.full_like(arch_idx, -1)
        )
        return phi, mix, tracked_idx

    def update_scores(
        self,
        phi_batch: torch.Tensor,
        mix_batch: torch.Tensor,
        tracked_idx: torch.Tensor,
        learnability_per_env: torch.Tensor,
    ) -> None:
        """Update archive learnability scores from per-env advantage variance / TD-error."""
        if self.strategy != "ability_upd":
            return

        # 1. Update replayed archive entries via EMA
        replayed = tracked_idx >= 0
        if replayed.any():
            r_idx = tracked_idx[replayed]
            r_scores = learnability_per_env[replayed]
            self.archive_scores[r_idx] = 0.7 * self.archive_scores[r_idx] + 0.3 * r_scores

        # 2. Insert highest-learnability fresh candidates over lowest-scoring archive entries
        fresh = ~replayed
        if fresh.any():
            f_scores = learnability_per_env[fresh]
            f_phi = phi_batch[fresh]
            f_mix = mix_batch[fresh]
            k = min(6, f_scores.numel())
            top_vals, top_i = torch.topk(f_scores, k=k)
            low_vals, low_i = torch.topk(self.archive_scores, k=k, largest=False)
            replace = top_vals > low_vals
            if replace.any():
                dst = low_i[replace]
                src = top_i[replace]
                self.archive_phi[dst] = f_phi[src]
                self.archive_mix[dst] = f_mix[src]
                self.archive_scores[dst] = top_vals[replace]
