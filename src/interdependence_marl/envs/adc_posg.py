"""Batched PyTorch Ability-Parameterized Partially Observable Stochastic Game (A-POSG).

Implements both collaboration topologies from Xiao et al. (CHI '24):
  1. Ability Combining (`mode="ability_combining"`): Shared goal where agents contribute
     strictly complementary Required Abilities (e.g., visual code identification +
     elevated dispenser actuation) and exchange Supportive Abilities.
  2. Ability Sharing (`mode="ability_sharing"`): Goal belongs privately to Agent 0
     (Ability Borrower), while Agent 1 (Ability Provider) supplies Required Abilities
     and must respect Borrower agency rather than performing unilateral overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

NUM_REQ_ACTIONS = 6   # 0: stay, 1: up, 2: down, 3: left, 4: right, 5: interact
NUM_SUPP_ACTIONS = 7  # 0: silent, 1..3: vocal symbols 0..2, 4..6: visual symbols 0..2
NUM_STATIONS = 3
OBS_DIM = 42
STATE_DIM = 51


@dataclass
class EnvConfig:
    grid_size: int = 5
    max_steps: int = 26
    mode: str = "ability_combining"  # "ability_combining" or "ability_sharing"
    interdependence_reward: bool = True
    supp_cost_borrower: float = 0.022
    supp_cost_provider: float = 0.006
    agency_bonus: float = 0.45
    override_penalty: float = 0.50
    wrong_item_penalty: float = 0.10
    extract_bonus: float = 0.55
    step_penalty: float = 0.005
    potential_scale: float = 0.05


class AbilityDiversePOSG:
    """Vectorized 2-agent Ability-Parameterized POSG running natively in PyTorch."""

    def __init__(
        self,
        num_envs: int,
        config: EnvConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.cfg = config or EnvConfig()
        self.device = torch.device(device)
        assert self.cfg.mode in ("ability_combining", "ability_sharing")

        g = self.cfg.grid_size - 1
        self.corners = torch.tensor(
            [[0, 0], [0, g], [g, 0], [g, g]], dtype=torch.long, device=self.device
        )
        mid = self.cfg.grid_size // 2
        self.checkout_pos = torch.tensor([mid, mid], dtype=torch.long, device=self.device)
        self.spawn_0 = torch.tensor([0, mid], dtype=torch.long, device=self.device)
        self.spawn_1 = torch.tensor([g, mid], dtype=torch.long, device=self.device)

        self.move_deltas = torch.tensor(
            [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]],
            dtype=torch.long,
            device=self.device,
        )

        # Persistent episode tensors
        self.phi = torch.ones((num_envs, 2, 4), dtype=torch.float32, device=self.device)
        self.pos = torch.zeros((num_envs, 2, 2), dtype=torch.long, device=self.device)
        self.station_pos = torch.zeros(
            (num_envs, NUM_STATIONS, 2), dtype=torch.long, device=self.device
        )
        self.station_vis_code = torch.zeros(
            (num_envs, NUM_STATIONS), dtype=torch.long, device=self.device
        )
        self.target_station = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.target_code = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.held_by = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.last_supp = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.last_reach_fail = torch.zeros((num_envs, 2), dtype=torch.float32, device=self.device)
        self.relay_msg = torch.zeros((num_envs, 2, 3), dtype=torch.float32, device=self.device)

        # Per-episode diagnostic accumulators
        self.ep_req_moves = torch.zeros((num_envs, 2), dtype=torch.float32, device=self.device)
        self.ep_supp_count = torch.zeros((num_envs, 2), dtype=torch.float32, device=self.device)
        self.supp_cost_scale: float = 1.0
        self.dynamic_supp_cost_borrower: float | None = None
        self.dynamic_override_penalty: float | None = None
        self.dynamic_agency_bonus: float | None = None

    def reset(
        self,
        phi: torch.Tensor | None = None,
        env_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset all or masked environments with optional new ability profiles `(B, 2, 4)`."""
        if env_mask is None:
            idx = torch.arange(self.num_envs, device=self.device)
        else:
            idx = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)

        n = idx.numel()
        if n == 0:
            return self._get_obs(), self._get_state()

        if phi is not None:
            if phi.shape[0] == self.num_envs:
                self.phi[idx] = phi[idx].clamp(0.0, 1.0)
            else:
                self.phi[idx] = phi.clamp(0.0, 1.0)

        self.pos[idx, 0] = self.spawn_0
        self.pos[idx, 1] = self.spawn_1

        # Sample 3 distinct corners per environment for the 3 stations
        rand_corners = torch.argsort(torch.rand((n, 4), device=self.device), dim=-1)[:, :3]
        self.station_pos[idx] = self.corners[rand_corners]

        # Random permutation of visual item codes {0, 1, 2} across the 3 stations
        perms = torch.argsort(torch.rand((n, NUM_STATIONS), device=self.device), dim=-1)
        self.station_vis_code[idx] = perms

        t_station = torch.randint(0, NUM_STATIONS, (n,), device=self.device)
        self.target_station[idx] = t_station
        self.target_code[idx] = perms[torch.arange(n, device=self.device), t_station]

        self.held_by[idx] = -1
        self.steps[idx] = 0
        self.last_supp[idx] = 0
        self.last_reach_fail[idx] = 0.0
        self.relay_msg[idx] = 0.0
        self.ep_req_moves[idx] = 0.0
        self.ep_supp_count[idx] = 0.0

        return self._get_obs(), self._get_state()

    def set_relay_messages(self, relay_msg: torch.Tensor) -> None:
        """Inject assistive cross-modal relay tokens `(B, 2, 3)` from a Communication Supporter."""
        self.relay_msg = relay_msg.to(self.device)

    def _potential(self) -> torch.Tensor:
        """Potential function for policy-invariant spatial reward shaping."""
        b_idx = torch.arange(self.num_envs, device=self.device)
        t_pos = self.station_pos[b_idx, self.target_station]  # (B, 2)
        d0_st = (self.pos[:, 0] - t_pos).abs().sum(dim=-1).float()
        d1_st = (self.pos[:, 1] - t_pos).abs().sum(dim=-1).float()

        # Distance of closest reach-capable collaborator to target station
        eff_d0 = d0_st + 8.0 * (1.0 - self.phi[:, 0, 2])
        eff_d1 = d1_st + 8.0 * (1.0 - self.phi[:, 1, 2])
        pre_extract_dist = torch.minimum(eff_d0, eff_d1)

        d0_co = (self.pos[:, 0] - self.checkout_pos).abs().sum(dim=-1).float()
        d1_co = (self.pos[:, 1] - self.checkout_pos).abs().sum(dim=-1).float()
        carrier_dist = torch.where(self.held_by == 0, d0_co, d1_co)
        if self.cfg.mode == "ability_sharing" and self.cfg.interdependence_reward:
            post_extract_dist = 0.6 * carrier_dist + 0.4 * d0_co
        else:
            post_extract_dist = carrier_dist

        dist = torch.where(self.held_by >= 0, post_extract_dist, pre_extract_dist)
        return -dist

    def step(
        self,
        a_req: torch.Tensor,
        a_supp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Execute joint composite actions `a_req` (B, 2) and `a_supp` (B, 2)."""
        b_idx = torch.arange(self.num_envs, device=self.device)
        phi_prev = self._potential()

        self.steps += 1
        self.last_supp = a_supp.clone()
        self.last_reach_fail.zero_()

        # Track supportive actions & physical effort for interdependence evaluation
        self.ep_supp_count += (a_supp > 0).float()

        # 1. Apply motor execution slip noise (1 - phi_motor) to movement actions 1..4
        is_move = (a_req >= 1) & (a_req <= 4)
        slip_prob = 0.45 * (1.0 - self.phi[:, :, 3])
        slipped = is_move & (torch.rand_like(slip_prob) < slip_prob)
        random_move = torch.randint(0, 5, a_req.shape, device=self.device)
        eff_req = torch.where(slipped, random_move, a_req)

        deltas = self.move_deltas[eff_req]  # (B, 2, 2)
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
        consensual_checkout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        unilateral_override = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Supportive effort costs (scaled by curriculum warmup or Primal-Dual multiplier)
        if self.cfg.mode == "ability_sharing":
            if self.cfg.interdependence_reward:
                b_cost = (
                    self.dynamic_supp_cost_borrower
                    if self.dynamic_supp_cost_borrower is not None
                    else self.supp_cost_scale * self.cfg.supp_cost_borrower
                )
                reward -= b_cost * (a_supp[:, 0] > 0).float()
                reward -= (
                    self.supp_cost_scale
                    * self.cfg.supp_cost_provider
                    * (a_supp[:, 1] > 0).float()
                )
            else:
                reward -= 0.0005 * (a_supp > 0).float().sum(dim=-1)
        else:
            reward -= (
                self.supp_cost_scale
                * self.cfg.supp_cost_provider
                * (a_supp > 0).float().sum(dim=-1)
            )

        # 2. Item extraction at stations (when no item is currently held)
        for i in range(2):
            can_try = (self.held_by == -1) & (~done) & (a_req[:, i] == 5)
            if not can_try.any():
                continue
            # Check if agent i stands on any station k
            at_st = (self.pos[:, i].unsqueeze(1) == self.station_pos).all(dim=-1)  # (B, 3)
            on_any = can_try & at_st.any(dim=-1)
            if not on_any.any():
                continue

            st_k = at_st.float().argmax(dim=-1)  # (B,)
            # Co-located/adjacent partner boosts effective reach (cooperative lift)
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

            failed = on_any & (~reach_ok)
            self.last_reach_fail[:, i] = failed.float()

            extracted = on_any & reach_ok
            is_correct = extracted & (st_k == self.target_station)
            is_wrong = extracted & (st_k != self.target_station)

            self.held_by = torch.where(is_correct, torch.full_like(self.held_by, i), self.held_by)
            reward = torch.where(is_correct, reward + self.cfg.extract_bonus, reward)

            reward = torch.where(is_wrong, reward - self.cfg.wrong_item_penalty, reward)
            done = done | is_wrong
            wrong_extract = wrong_extract | is_wrong

        # 3. Item handoff between adjacent/co-located collaborators
        can_handoff = (
            (self.held_by >= 0)
            & (~done)
            & ((self.pos[:, 0] - self.pos[:, 1]).abs().sum(dim=-1) <= 1)
        )
        # In ability_sharing, if Borrower (0) interacts near carrier Provider (1), item transfers to 0
        handoff_to_0 = can_handoff & (self.held_by == 1) & (a_req[:, 0] == 5)
        self.held_by = torch.where(handoff_to_0, torch.zeros_like(self.held_by), self.held_by)

        # 4. Checkout at center cell (2, 2)
        for i in range(2):
            at_co = (self.pos[:, i] == self.checkout_pos).all(dim=-1)
            checking_out = (self.held_by == i) & (~done) & at_co & (a_req[:, i] == 5)
            if not checking_out.any():
                continue

            done = done | checking_out
            success = success | checking_out
            reward = torch.where(checking_out, reward + 1.0, reward)

            if self.cfg.mode == "ability_sharing":
                borrower_dist = (self.pos[:, 0] - self.checkout_pos).abs().sum(dim=-1)
                is_cons = checking_out & (borrower_dist <= 1)
                is_unilat = checking_out & (borrower_dist > 1)
                consensual_checkout = consensual_checkout | is_cons
                unilateral_override = unilateral_override | is_unilat

                if self.cfg.interdependence_reward:
                    ag_bon = (
                        self.dynamic_agency_bonus
                        if self.dynamic_agency_bonus is not None
                        else self.cfg.agency_bonus
                    )
                    ov_pen = (
                        self.dynamic_override_penalty
                        if self.dynamic_override_penalty is not None
                        else self.cfg.override_penalty
                    )
                    reward = torch.where(is_cons, reward + ag_bon, reward)
                    reward = torch.where(is_unilat, reward - ov_pen, reward)

        # Potential-based shaping on non-terminal transitions
        phi_next = self._potential()
        shaping = self.cfg.potential_scale * (0.99 * phi_next - phi_prev)
        reward = torch.where(done, reward, reward + shaping)

        timeout = self.steps >= self.cfg.max_steps
        done = done | timeout

        # Clear relay messages before generating fresh observations (unless set externally next step)
        obs = self._get_obs()
        state = self._get_state()
        self.relay_msg.zero_()

        # Contribution symmetry index on episode steps: 1 - |m0 - m1| / (m0 + m1 + eps)
        m0 = self.ep_req_moves[:, 0] + self.ep_supp_count[:, 0]
        m1 = self.ep_req_moves[:, 1] + self.ep_supp_count[:, 1]
        csi = 1.0 - (m0 - m1).abs() / (m0 + m1 + 1e-6)

        info = {
            "success": success,
            "wrong_extract": wrong_extract,
            "timeout": timeout & (~success) & (~wrong_extract),
            "consensual_checkout": consensual_checkout,
            "unilateral_override": unilateral_override,
            "borrower_supp_count": self.ep_supp_count[:, 0].clone(),
            "provider_supp_count": self.ep_supp_count[:, 1].clone(),
            "total_supp_count": self.ep_supp_count.sum(dim=-1),
            "csi": csi,
            "target_code": self.target_code.clone(),
            "b_idx": b_idx,
        }
        return obs, state, reward, done, info

    def _get_obs(self) -> torch.Tensor:
        """Construct ability-filtered partial observations `(B, 2, OBS_DIM=42)`."""
        b = self.num_envs
        scale = float(self.cfg.grid_size - 1)
        obs_list = []

        st_coords = (self.station_pos.float() / scale).reshape(b, 6)  # (B, 6)
        st_codes_oh = F.one_hot(self.station_vis_code, num_classes=NUM_STATIONS).float()  # (B, 3, 3)
        st_codes_flat = st_codes_oh.reshape(b, 9)
        target_oh = F.one_hot(self.target_code, num_classes=NUM_STATIONS).float()  # (B, 3)
        rem_time = ((self.cfg.max_steps - self.steps).float() / self.cfg.max_steps).unsqueeze(-1)
        mode_flag = torch.full(
            (b, 1),
            1.0 if self.cfg.mode == "ability_sharing" else 0.0,
            dtype=torch.float32,
            device=self.device,
        )

        for i in range(2):
            p = 1 - i
            own_pos = self.pos[:, i].float() / scale  # (B, 2)
            role_oh = torch.zeros((b, 2), dtype=torch.float32, device=self.device)
            role_oh[:, i] = 1.0
            own_phi = self.phi[:, i]  # (B, 4): (vis, aud, reach, motor)

            vis_i = own_phi[:, 0:1]  # (B, 1)
            aud_i = own_phi[:, 1:2]  # (B, 1)

            # Partner position sensed via max(vision, 0.7 * spatial_audio)
            loc_clarity = torch.maximum(vis_i, 0.7 * aud_i)
            partner_true = self.pos[:, p].float() / scale
            partner_pos = loc_clarity * partner_true + (1.0 - loc_clarity) * 0.5

            # Station visual labels gated strictly by own visual capability vis_i
            vis_noise = torch.randn((b, 9), device=self.device) * 0.55 * (1.0 - vis_i)
            vis_codes_obs = torch.where(
                vis_i > 0.02,
                vis_i * st_codes_flat + vis_noise,
                torch.zeros_like(st_codes_flat),
            )

            # Goal visibility:
            # - ability_combining: shown on visual board -> gated by vis_i
            # - ability_sharing: privately owned by Borrower (Agent 0); strictly hidden from Agent 1
            if self.cfg.mode == "ability_combining":
                goal_noise = torch.randn((b, 3), device=self.device) * 0.45 * (1.0 - vis_i)
                goal_obs = torch.where(
                    vis_i > 0.02,
                    vis_i * target_oh + goal_noise,
                    torch.zeros_like(target_oh),
                )
            else:
                if i == 0:
                    goal_obs = target_oh
                else:
                    goal_obs = torch.zeros_like(target_oh)

            # Incoming vocal signal (a_supp in {1, 2, 3}) gated by sender_aud * receiver_aud,
            # plus assistive TTS relay (delivered via receiver's auditory channel aud_i)
            p_supp = self.last_supp[:, p]
            is_vocal = (p_supp >= 1) & (p_supp <= 3)
            vocal_sym = (p_supp - 1).clamp(0, 2)
            vocal_oh = F.one_hot(vocal_sym, num_classes=3).float() * is_vocal.unsqueeze(-1).float()
            vocal_gain = self.phi[:, p, 1:2] * aud_i
            relay_obs = self.relay_msg[:, i]  # (B, 3)
            vocal_obs = (vocal_oh * vocal_gain + relay_obs * aud_i).clamp(0.0, 1.0)

            # Incoming visual/deictic gesture (a_supp in {4, 5, 6}) gated by sender_vis * receiver_vis,
            # plus assistive Live-Caption relay (delivered via receiver's visual channel vis_i)
            is_gest = (p_supp >= 4) & (p_supp <= 6)
            gest_sym = (p_supp - 4).clamp(0, 2)
            gest_oh = F.one_hot(gest_sym, num_classes=3).float() * is_gest.unsqueeze(-1).float()
            gest_gain = self.phi[:, p, 0:1] * vis_i
            gest_obs = (gest_oh * gest_gain + relay_obs * vis_i).clamp(0.0, 1.0)

            self_carries = (self.held_by == i).float().unsqueeze(-1)
            partner_carries = (self.held_by == p).float().unsqueeze(-1)
            partner_reach_fail = self.last_reach_fail[:, p].unsqueeze(-1)

            obs_i = torch.cat(
                [
                    own_pos,             # 2  [0:2]
                    role_oh,             # 2  [2:4]
                    own_phi,             # 4  [4:8]
                    st_coords,           # 6  [8:14]
                    partner_pos,         # 2  [14:16]
                    vis_codes_obs,       # 9  [16:25]
                    goal_obs,            # 3  [25:28]
                    vocal_obs,           # 3  [28:31]
                    gest_obs,            # 3  [31:34]
                    relay_obs,           # 3  [34:37]
                    self_carries,        # 1  [37:38]
                    partner_carries,     # 1  [38:39]
                    rem_time,            # 1  [39:40]
                    partner_reach_fail,  # 1  [40:41]
                    mode_flag,           # 1  [41:42]
                ],
                dim=-1,
            )
            obs_list.append(obs_i)

        return torch.stack(obs_list, dim=1)  # (B, 2, 42)

    def _get_state(self) -> torch.Tensor:
        """Construct privileged global state `(B, STATE_DIM=51)` for centralized critic."""
        b = self.num_envs
        scale = float(self.cfg.grid_size - 1)
        pos_flat = (self.pos.float() / scale).reshape(b, 4)
        st_coords = (self.station_pos.float() / scale).reshape(b, 6)
        st_codes = F.one_hot(self.station_vis_code, num_classes=NUM_STATIONS).float().reshape(b, 9)
        t_code = F.one_hot(self.target_code, num_classes=NUM_STATIONS).float()
        t_stat = F.one_hot(self.target_station, num_classes=NUM_STATIONS).float()
        phis = self.phi.reshape(b, 8)
        held_oh = F.one_hot((self.held_by + 1).clamp(0, 2), num_classes=3).float()
        supp_oh = F.one_hot(self.last_supp, num_classes=NUM_SUPP_ACTIONS).float().reshape(b, 14)
        rem_time = ((self.cfg.max_steps - self.steps).float() / self.cfg.max_steps).unsqueeze(-1)
        return torch.cat(
            [pos_flat, st_coords, st_codes, t_code, t_stat, phis, held_oh, supp_oh, rem_time],
            dim=-1,
        )
