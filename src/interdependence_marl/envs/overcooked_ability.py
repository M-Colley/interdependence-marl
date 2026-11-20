"""Tier 2 Benchmark: Procedural `OvercookedV2-Ability` Multi-Stage Cooperative Kitchen.

Extends `AbilityDiversePOSG` with procedural kitchen layouts (inspired by Bulling's
Overcooked Generalisation Challenge, TMLR 2025, and OvercookedV2) and a 3-stage recipe:
  Stage 0: Identify customer soup order (Tomato/Onion/Mushroom, gated by `vis`) and
           fetch raw ingredient from elevated pantry (gated by `reach`).
  Stage 1: Deliver raw ingredient to procedurally placed Cooking Pot (`pot_pos`) and simmer.
  Stage 2: Plate cooked soup and serve at Serving Window `(2, 2)`.
"""

from __future__ import annotations

import torch

from interdependence_marl.envs.adc_posg import AbilityDiversePOSG, EnvConfig


class OvercookedAbilityEnv(AbilityDiversePOSG):
    """Batched procedural Overcooked-Ability environment (`OBS_DIM=46, STATE_DIM=54`)."""

    def __init__(
        self,
        num_envs: int,
        config: EnvConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        cfg = config or EnvConfig(max_steps=32, mode="ability_combining")
        super().__init__(num_envs=num_envs, config=cfg, device=device)

        # Candidate interior stove/pot locations (varied procedurally per episode like OGC)
        self.pot_candidates = torch.tensor(
            [[1, 1], [1, 3], [3, 1], [3, 3]], dtype=torch.long, device=self.device
        )
        self.pot_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.pot_cooked = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def reset(
        self,
        phi: torch.Tensor | None = None,
        env_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_mask is None:
            idx = torch.arange(self.num_envs, device=self.device)
        else:
            idx = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)

        if idx.numel() > 0:
            pot_idx = torch.randint(0, 4, (idx.numel(),), device=self.device)
            self.pot_pos[idx] = self.pot_candidates[pot_idx]
            self.pot_cooked[idx] = False

        return super().reset(phi=phi, env_mask=env_mask)

    def _potential(self) -> torch.Tensor:
        b_idx = torch.arange(self.num_envs, device=self.device)
        t_pos = self.station_pos[b_idx, self.target_station]

        d0_st = (self.pos[:, 0] - t_pos).abs().sum(dim=-1).float()
        d1_st = (self.pos[:, 1] - t_pos).abs().sum(dim=-1).float()
        eff_d0 = d0_st + 8.0 * (1.0 - self.phi[:, 0, 2])
        eff_d1 = d1_st + 8.0 * (1.0 - self.phi[:, 1, 2])
        stage0_dist = torch.minimum(eff_d0, eff_d1)

        d0_pot = (self.pos[:, 0] - self.pot_pos).abs().sum(dim=-1).float()
        d1_pot = (self.pos[:, 1] - self.pot_pos).abs().sum(dim=-1).float()
        stage1_dist = torch.where(self.held_by == 0, d0_pot, d1_pot)

        d0_co = (self.pos[:, 0] - self.checkout_pos).abs().sum(dim=-1).float()
        d1_co = (self.pos[:, 1] - self.checkout_pos).abs().sum(dim=-1).float()
        stage2_dist = torch.where(self.held_by == 0, d0_co, d1_co)

        post_fetch = torch.where(self.pot_cooked, stage2_dist, stage1_dist + 2.0)
        dist = torch.where(self.held_by >= 0, post_fetch, stage0_dist + 4.0)
        return -dist

    def step(
        self,
        a_req: torch.Tensor,
        a_supp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b_idx = torch.arange(self.num_envs, device=self.device)
        phi_prev = self._potential()

        self.steps += 1
        self.last_supp = a_supp.clone()
        self.last_reach_fail.zero_()
        self.ep_supp_count += (a_supp > 0).float()

        is_move = (a_req >= 1) & (a_req <= 4)
        slip_prob = 0.45 * (1.0 - self.phi[:, :, 3])
        slipped = is_move & (torch.rand_like(slip_prob) < slip_prob)
        random_move = torch.randint(0, 5, a_req.shape, device=self.device)
        eff_req = torch.where(slipped, random_move, a_req)

        deltas = self.move_deltas[eff_req]
        new_pos = (self.pos + deltas).clamp(0, self.cfg.grid_size - 1)
        moved = (new_pos != self.pos).any(dim=-1).float()
        self.pos = new_pos
        self.ep_req_moves += moved + (a_req == 5).float()

        reward = torch.full(
            (self.num_envs,), -self.cfg.step_penalty, dtype=torch.float32, device=self.device
        )
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        wrong_extract = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        reward -= (
            self.supp_cost_scale * self.cfg.supp_cost_provider * (a_supp > 0).float().sum(dim=-1)
        )

        # Stage 0: Fetch raw ingredient from elevated pantry
        for i in range(2):
            can_try = (self.held_by == -1) & (~done) & (a_req[:, i] == 5)
            if not can_try.any():
                continue
            at_st = (self.pos[:, i].unsqueeze(1) == self.station_pos).all(dim=-1)
            on_any = can_try & at_st.any(dim=-1)
            if not on_any.any():
                continue

            st_k = at_st.float().argmax(dim=-1)
            partner = 1 - i
            partner_near = (
                (self.pos[:, partner] - self.pos[:, i]).abs().sum(dim=-1) <= 1
            ) & (a_req[:, partner] == 5)
            eff_reach = torch.where(
                partner_near,
                (self.phi[:, i, 2] + self.phi[:, partner, 2]).clamp(0.0, 1.0),
                self.phi[:, i, 2],
            )
            reach_ok = torch.rand(self.num_envs, device=self.device) < eff_reach
            self.last_reach_fail[:, i] = (on_any & (~reach_ok)).float()

            extracted = on_any & reach_ok
            is_correct = extracted & (st_k == self.target_station)
            is_wrong = extracted & (st_k != self.target_station)

            self.held_by = torch.where(is_correct, torch.full_like(self.held_by, i), self.held_by)
            reward = torch.where(is_correct, reward + 0.40, reward)
            reward = torch.where(is_wrong, reward - self.cfg.wrong_item_penalty, reward)
            done = done | is_wrong
            wrong_extract = wrong_extract | is_wrong

        # Allow ingredient/soup counter handoff between adjacent chefs
        adjacent = ((self.pos[:, 0] - self.pos[:, 1]).abs().sum(dim=-1) <= 1) & (~done)
        pass_0_to_1 = adjacent & (self.held_by == 0) & (a_req[:, 1] == 5)
        pass_1_to_0 = adjacent & (self.held_by == 1) & (a_req[:, 0] == 5)
        self.held_by = torch.where(pass_0_to_1, torch.ones_like(self.held_by), self.held_by)
        self.held_by = torch.where(pass_1_to_0, torch.zeros_like(self.held_by), self.held_by)

        # Stage 1: Simmer raw ingredient at procedural Cooking Pot
        just_cooked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(2):
            at_pot = (self.pos[:, i] == self.pot_pos).all(dim=-1)
            cook_act = (self.held_by == i) & (~self.pot_cooked) & (~done) & at_pot & (a_req[:, i] == 5)
            if cook_act.any():
                self.pot_cooked = self.pot_cooked | cook_act
                just_cooked = just_cooked | cook_act
                reward = torch.where(cook_act, reward + 0.45, reward)

        # Stage 2: Serve cooked soup at Serving Window (2, 2)
        for i in range(2):
            at_serve = (self.pos[:, i] == self.checkout_pos).all(dim=-1)
            serve_act = (
                (self.held_by == i)
                & self.pot_cooked
                & (~just_cooked)
                & (~done)
                & at_serve
                & (a_req[:, i] == 5)
            )
            if serve_act.any():
                done = done | serve_act
                success = success | serve_act
                reward = torch.where(serve_act, reward + 1.0, reward)

        phi_next = self._potential()
        shaping = self.cfg.potential_scale * (0.99 * phi_next - phi_prev)
        reward = torch.where(done, reward, reward + shaping)

        timeout = self.steps >= self.cfg.max_steps
        done = done | timeout

        obs = self._get_obs()
        state = self._get_state()
        self.relay_msg.zero_()

        m0 = self.ep_req_moves[:, 0] + self.ep_supp_count[:, 0]
        m1 = self.ep_req_moves[:, 1] + self.ep_supp_count[:, 1]
        csi = 1.0 - (m0 - m1).abs() / (m0 + m1 + 1e-6)
        zeros_bool = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        info = {
            "success": success,
            "wrong_extract": wrong_extract,
            "timeout": timeout & (~success) & (~wrong_extract),
            "consensual_checkout": zeros_bool,
            "unilateral_override": zeros_bool,
            "borrower_supp_count": self.ep_supp_count[:, 0].clone(),
            "provider_supp_count": self.ep_supp_count[:, 1].clone(),
            "total_supp_count": self.ep_supp_count.sum(dim=-1),
            "csi": csi,
            "target_code": self.target_code.clone(),
            "b_idx": b_idx,
        }
        return obs, state, reward, done, info

    def _get_obs(self) -> torch.Tensor:
        base_obs = super()._get_obs()  # (B, 2, 42)
        scale = float(self.cfg.grid_size - 1)
        pot_norm = (self.pot_pos.float() / scale).unsqueeze(1).expand(-1, 2, -1)  # (B, 2, 2)
        own_pos = base_obs[:, :, 0:2]
        delta_pot = pot_norm - own_pos  # (B, 2, 2)
        at_pot = (delta_pot.abs().sum(dim=-1, keepdim=True) < 0.05).float()  # (B, 2, 1)
        cooked_flag = self.pot_cooked.float().view(-1, 1, 1).expand(-1, 2, 1)  # (B, 2, 1)
        return torch.cat([base_obs, delta_pot, at_pot, cooked_flag], dim=-1)  # (B, 2, 46)

    def _get_state(self) -> torch.Tensor:
        base_state = super()._get_state()  # (B, 51)
        scale = float(self.cfg.grid_size - 1)
        pot_norm = self.pot_pos.float() / scale  # (B, 2)
        cooked_flag = self.pot_cooked.float().unsqueeze(-1)  # (B, 1)
        return torch.cat([base_state, pot_norm, cooked_flag], dim=-1)  # (B, 54)
