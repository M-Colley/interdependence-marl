"""Tier 3 Benchmark: Continuous 2D Biomechanical Shared Control & Variational Empowerment.

Inspired by Bulling's `iAssistADL` project & Charaja, ..., Bulling, et al. (BioRob 2024)
on reinforcement learning for musculoskeletal reaching under neurodegenerative movement
disorders (Parkinsonian tremor, cerebellar ataxia dysmetria, weakness/ROM constraints).

Compares:
  1. `unassisted`: Impaired human controller acting solo (high agency, high tremor/error).
  2. `task_takeover`: Standard shared-autonomy RL maximizing task return alone (stiff force
     override -> high reaching score on clean gaze, but crushes user empowerment `I(A_u; S' | S)`).
  3. `adc_lagrangian`: Primal-Dual Interdependent RL maximizing task success subject to a
     Barber-Agakov Variational Empowerment constraint `I_q(A_u; S_{t+3} | S_t) >= tau_emp`
     (learns transparent tremor damping while preserving voluntary user control authority).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn

from interdependence_marl.evaluation.empowerment import VariationalEmpowermentNet


@dataclass
class BioEnvConfig:
    dt: float = 0.08
    max_steps: int = 35
    target_radius: float = 0.15
    tremor_freq_hz: float = 5.0
    damping: float = 2.1
    assist_gain: float = 2.25
    # Impaired user profile: (gaze_snr, rom_limit, tremor_free, strength)
    gaze_snr: float = 0.65
    rom_limit: float = 0.65
    tremor_free: float = 0.20   # Severe 5Hz tremor + dysmetria (0.80 amplitude)
    strength: float = 0.55


class ContinuousBiomechanicalEnv:
    """Batched 2D continuous-physics reaching task with pathological tremor & shared control."""

    def __init__(
        self,
        num_envs: int,
        cfg: BioEnvConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.cfg = cfg or BioEnvConfig()
        self.device = torch.device(device)

        self.pos = torch.zeros((num_envs, 2), device=self.device)
        self.vel = torch.zeros((num_envs, 2), device=self.device)
        self.u_vol = torch.zeros((num_envs, 2), device=self.device)
        self.target = torch.zeros((num_envs, 2), device=self.device)
        self.gaze_cue = torch.zeros((num_envs, 2), device=self.device)
        self.phase = torch.zeros((num_envs, 2), device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def reset(self) -> torch.Tensor:
        angles = torch.rand(self.num_envs, device=self.device) * (2.0 * math.pi)
        radii = 0.75 + 0.20 * torch.rand(self.num_envs, device=self.device)
        self.target = torch.stack([radii * angles.cos(), radii * angles.sin()], dim=-1)

        # Ambiguous dining-table eye-gaze cue: noisy + occasional nearby distractor object (30%)
        gaze_noise = torch.randn_like(self.target) * (0.35 * (1.0 - self.cfg.gaze_snr))
        distractor = (torch.rand((self.num_envs, 1), device=self.device) < 0.30).float()
        orth = torch.stack([-self.target[:, 1], self.target[:, 0]], dim=-1) * 0.45
        self.gaze_cue = self.target + gaze_noise + distractor * orth

        self.pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.vel = torch.zeros((self.num_envs, 2), device=self.device)
        self.u_vol = torch.zeros((self.num_envs, 2), device=self.device)
        self.phase = torch.rand((self.num_envs, 2), device=self.device) * (2.0 * math.pi)
        self.steps.zero_()
        return self._get_assist_obs()

    def user_voluntary_action(self, exploratory_noise: float = 0.35) -> torch.Tensor:
        """Simulated human voluntary PD motor command toward true private `self.target`."""
        err = self.target - self.pos
        cmd = 1.8 * err - 0.35 * self.vel
        if exploratory_noise > 0.0:
            cmd = cmd + exploratory_noise * torch.randn_like(cmd)
        return torch.tanh(cmd)

    def _get_assist_obs(self) -> torch.Tensor:
        """Assistive controller observes `[pos(2), vel(2), gaze_cue(2), gaze_cue - pos(2), t_rem(1)]` (9D)."""
        t_rem = ((self.cfg.max_steps - self.steps).float() / self.cfg.max_steps).unsqueeze(-1)
        return torch.cat(
            [self.pos, self.vel, self.gaze_cue, self.gaze_cue - self.pos, t_rem], dim=-1
        )

    def kinematic_state(self) -> torch.Tensor:
        """4D limb kinematic state `[pos(2), vel(2)]` used for Variational Empowerment `I(A_u; S' | S)`."""
        return torch.cat([self.pos, self.vel], dim=-1)

    def step(
        self, a_user: torch.Tensor, a_assist: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        dt = self.cfg.dt
        prev_dist = (self.pos - self.target).norm(dim=-1)
        self.steps += 1

        # First-order muscle activation filter
        self.u_vol = 0.65 * self.u_vol + 0.35 * a_user.clamp(-1.0, 1.0)

        # Range-of-motion saturation + muscle weakness
        radial = self.pos.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rom_atten = torch.sigmoid(6.0 * (self.cfg.rom_limit - radial))
        f_vol = self.cfg.strength * (0.35 + 0.65 * rom_atten) * self.u_vol

        # Pathological 5Hz oscillatory tremor + signal-dependent dysmetria noise
        t_sec = self.steps.float().unsqueeze(-1) * dt
        omega = 2.0 * math.pi * self.cfg.tremor_freq_hz
        trem_amp = 1.15 * (1.0 - self.cfg.tremor_free)
        osc = torch.cat(
            [
                torch.sin(omega * t_sec + self.phase[:, 0:1]),
                torch.cos(omega * t_sec + self.phase[:, 1:2]),
            ],
            dim=-1,
        )
        sdn = 0.35 * (1.0 - self.cfg.tremor_free) * self.u_vol.norm(dim=-1, keepdim=True)
        f_pathological = trem_amp * osc + sdn * torch.randn_like(self.pos)

        f_assist = self.cfg.assist_gain * a_assist.clamp(-1.0, 1.0)
        acc = f_vol + f_pathological + f_assist - self.cfg.damping * self.vel

        self.vel = (self.vel + dt * acc).clamp(-2.5, 2.5)
        self.pos = (self.pos + dt * self.vel).clamp(-1.4, 1.4)

        dist = (self.pos - self.target).norm(dim=-1)
        in_target = dist <= self.cfg.target_radius
        speed = self.vel.norm(dim=-1)
        stabilized = in_target & (speed < 0.55)

        # Task reward: potential progress + target dwell + stability
        task_rew = (
            1.5 * (prev_dist - dist)
            + 0.35 * in_target.float()
            + 0.55 * stabilized.float()
        )
        done = self.steps >= self.cfg.max_steps
        info = {
            "dist": dist,
            "in_target": in_target,
            "stabilized": stabilized,
            "assist_mag": a_assist.norm(dim=-1),
            "speed": speed,
        }
        return self._get_assist_obs(), task_rew, done, info


class ContinuousAssistActorCritic(nn.Module):
    """Gaussian PPO policy for continuous 2D assistive force control."""

    def __init__(self, obs_dim: int = 9, act_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.actor(obs))
        std = self.log_std.clamp(-2.5, 0.5).exp()
        if deterministic:
            raw = mean
        else:
            raw = mean + std * torch.randn_like(mean)
        logp = -0.5 * (((raw - mean) / std).pow(2) + 2.0 * self.log_std.clamp(-2.5, 0.5) + math.log(2.0 * math.pi))
        val = self.critic(obs).squeeze(-1)
        return raw.clamp(-1.0, 1.0), logp.sum(dim=-1), val

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.actor(obs))
        lstd = self.log_std.clamp(-2.5, 0.5)
        std = lstd.exp()
        logp = -0.5 * (((actions - mean) / std).pow(2) + 2.0 * lstd + math.log(2.0 * math.pi))
        ent = 0.5 * (1.0 + math.log(2.0 * math.pi) + 2.0 * lstd).sum()
        val = self.critic(obs).squeeze(-1)
        return logp.sum(dim=-1), ent.expand(obs.shape[0]), val


def train_and_eval_tier3_biomechanical(
    num_updates: int = 90,
    num_envs: int = 192,
    seed: int = 77,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Train `Task-Only Shared Autonomy` vs. `Primal-Dual Empowerment-Constrained Assist`."""
    dev = torch.device(device)
    cfg = BioEnvConfig()
    k_horizon = 3

    results: dict[str, object] = {}
    sample_trajectories: dict[str, dict[str, list[list[float]]]] = {}

    # 1. Evaluate Unassisted Baseline
    torch.manual_seed(seed)
    env_u = ContinuousBiomechanicalEnv(num_envs=256, cfg=cfg, device=dev)
    emp_probe_u = VariationalEmpowermentNet(state_dim=4, action_dim=2).to(dev)
    opt_q_u = torch.optim.Adam(emp_probe_u.parameters(), lr=3e-3)
    # Fit variational empowerment probe on unassisted trajectories
    for _ in range(35):
        env_u.reset()
        s_hist, a_hist = [], []
        for _ in range(cfg.max_steps):
            s_hist.append(env_u.kinematic_state())
            au = env_u.user_voluntary_action(exploratory_noise=0.50)
            a_hist.append(au)
            env_u.step(au, torch.zeros_like(au))
        s_t = torch.stack(s_hist[:-k_horizon]).reshape(-1, 4)
        s_tk = torch.stack(s_hist[k_horizon:]).reshape(-1, 4)
        a_t = torch.stack(a_hist[:-k_horizon]).reshape(-1, 2)
        loss_q = -emp_probe_u.log_prob(s_t, s_tk, a_t).mean()
        opt_q_u.zero_grad()
        loss_q.backward()
        opt_q_u.step()

    with torch.no_grad():
        torch.manual_seed(seed + 999)
        env_u.reset()
        s_hist, a_hist, stab_last, dist_last = [], [], None, None
        traj_u = [env_u.pos[0].tolist()]
        target_sample = env_u.target[0].tolist()
        for _ in range(cfg.max_steps):
            s_hist.append(env_u.kinematic_state())
            au = env_u.user_voluntary_action(exploratory_noise=0.40)
            a_hist.append(au)
            _, _, _, inf = env_u.step(au, torch.zeros_like(au))
            traj_u.append(env_u.pos[0].tolist())
            stab_last = inf["in_target"]
            dist_last = inf["dist"]
        s_t = torch.stack(s_hist[:-k_horizon]).reshape(-1, 4)
        s_tk = torch.stack(s_hist[k_horizon:]).reshape(-1, 4)
        a_t = torch.stack(a_hist[:-k_horizon]).reshape(-1, 2)
        emp_u = float(emp_probe_u.estimate_mi(s_t, s_tk, a_t, prior_std=0.55).mean().item())
        results["Unassisted Impaired"] = {
            "target_reach_rate": float(stab_last.float().mean().item()),
            "final_dist": float(dist_last.mean().item()),
            "user_empowerment_nats": emp_u,
            "assist_force_norm": 0.0,
        }
        sample_trajectories["target"] = {"xy": [target_sample]}
        sample_trajectories["Unassisted Impaired"] = {"xy": traj_u}

    # 2. Train Task-Only Takeover vs. Primal-Dual Empowerment-Preserving Assist
    for mode_name, use_emp_constraint in [
        ("Task-Only Shared Autonomy", False),
        ("ADC-Lagrangian Assist (Ours)", True),
    ]:
        torch.manual_seed(seed)
        env = ContinuousBiomechanicalEnv(num_envs=num_envs, cfg=cfg, device=dev)
        policy = ContinuousAssistActorCritic().to(dev)
        emp_net = VariationalEmpowermentNet(state_dim=4, action_dim=2).to(dev)
        opt_pi = torch.optim.Adam(policy.parameters(), lr=2.2e-3)
        opt_emp = torch.optim.Adam(emp_net.parameters(), lr=3.0e-3)

        nu_emp = 0.25
        tau_emp = 0.66  # Target minimum user empowerment in nats

        for upd in range(num_updates):
            obs = env.reset()
            T = cfg.max_steps
            b_obs = torch.zeros((T, num_envs, 9), device=dev)
            b_kin = torch.zeros((T + 1, num_envs, 4), device=dev)
            b_au = torch.zeros((T, num_envs, 2), device=dev)
            b_aa = torch.zeros((T, num_envs, 2), device=dev)
            b_logp = torch.zeros((T, num_envs), device=dev)
            b_val = torch.zeros((T, num_envs), device=dev)
            b_rew = torch.zeros((T, num_envs), device=dev)

            b_kin[0] = env.kinematic_state()
            for t in range(T):
                b_obs[t] = obs
                au = env.user_voluntary_action(exploratory_noise=0.40)
                with torch.no_grad():
                    aa, lp, val = policy.act(obs)
                obs, r_task, _, inf = env.step(au, aa)
                b_kin[t + 1] = env.kinematic_state()
                b_au[t] = au
                b_aa[t] = aa
                b_logp[t] = lp
                b_val[t] = val
                # In Task-Only, naive shared autonomy optimizes proxy gaze-reaching without agency regularizer
                if not use_emp_constraint:
                    gaze_dist = (env.pos - env.gaze_cue).norm(dim=-1)
                    b_rew[t] = -0.8 * gaze_dist + 0.5 * (gaze_dist < cfg.target_radius).float()
                else:
                    b_rew[t] = r_task - 0.015 * aa.pow(2).sum(dim=-1)

            # Update Variational Empowerment inverse-dynamics estimator q_xi(a_u | s_t, s_{t+k})
            s_now = b_kin[: T - k_horizon].reshape(-1, 4).detach()
            s_fut = b_kin[k_horizon:T].reshape(-1, 4).detach()
            a_usr = b_au[: T - k_horizon].reshape(-1, 2).detach()
            for _ in range(3):
                q_loss = -emp_net.log_prob(s_now, s_fut, a_usr).mean()
                opt_emp.zero_grad()
                q_loss.backward()
                opt_emp.step()

            with torch.no_grad():
                mi_steps = emp_net.estimate_mi(s_now, s_fut, a_usr, prior_std=0.55).reshape(
                    T - k_horizon, num_envs
                )
                mean_mi = float(mi_steps.mean().item())

            if use_emp_constraint:
                # Augment step rewards with Lagrangian empowerment bonus + penalize opposing user command
                align_cos = (b_au * b_aa).sum(dim=-1)
                b_rew[: T - k_horizon] += nu_emp * (mi_steps - tau_emp) + 0.12 * align_cos[: T - k_horizon]
                # Projected dual ascent on nu_emp
                if upd >= 20:
                    nu_emp = min(0.85, max(0.0, nu_emp + 0.06 * (tau_emp - mean_mi)))

            # GAE & Clipped PPO update
            adv = torch.zeros_like(b_rew)
            gae = torch.zeros(num_envs, device=dev)
            for t in reversed(range(T)):
                vn = 0.0 if t == T - 1 else b_val[t + 1]
                delta = b_rew[t] + 0.98 * vn - b_val[t]
                gae = delta + 0.98 * 0.95 * gae
                adv[t] = gae
            ret = adv + b_val
            adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)

            flat_obs = b_obs.reshape(-1, 9)
            flat_act = b_aa.reshape(-1, 2)
            flat_old_lp = b_logp.reshape(-1)
            flat_adv = adv_n.reshape(-1)
            flat_ret = ret.reshape(-1)

            for _ in range(4):
                lp_new, ent, v_new = policy.evaluate(flat_obs, flat_act)
                ratio = (lp_new - flat_old_lp).exp()
                s1 = ratio * flat_adv
                s2 = ratio.clamp(0.8, 1.2) * flat_adv
                loss = -torch.min(s1, s2).mean() + 0.5 * (v_new - flat_ret).pow(2).mean() - 0.005 * ent.mean()
                opt_pi.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt_pi.step()

        # Fit a fresh, held-out evaluation VariationalEmpowermentNet for fair metric comparison
        torch.manual_seed(seed)
        eval_env = ContinuousBiomechanicalEnv(num_envs=256, cfg=cfg, device=dev)
        eval_q = VariationalEmpowermentNet(state_dim=4, action_dim=2).to(dev)
        opt_eq = torch.optim.Adam(eval_q.parameters(), lr=3e-3)
        for _ in range(35):
            obs_e = eval_env.reset()
            sh, ah = [], []
            for _ in range(cfg.max_steps):
                sh.append(eval_env.kinematic_state())
                au = eval_env.user_voluntary_action(exploratory_noise=0.50)
                with torch.no_grad():
                    aa, _, _ = policy.act(obs_e, deterministic=True)
                ah.append(au)
                obs_e, _, _, _ = eval_env.step(au, aa)
            st = torch.stack(sh[:-k_horizon]).reshape(-1, 4)
            stk = torch.stack(sh[k_horizon:]).reshape(-1, 4)
            at = torch.stack(ah[:-k_horizon]).reshape(-1, 2)
            lq = -eval_q.log_prob(st, stk, at).mean()
            opt_eq.zero_grad()
            lq.backward()
            opt_eq.step()

        # Final Evaluation on identical episode seed `(seed + 999)` so trajectory overlay shares target[0]
        torch.manual_seed(seed + 999)
        obs_e = eval_env.reset()
        sh, ah, f_mags = [], [], []
        traj_mode = [eval_env.pos[0].tolist()]
        in_targ_final, dist_final = None, None
        with torch.no_grad():
            for _ in range(cfg.max_steps):
                sh.append(eval_env.kinematic_state())
                au = eval_env.user_voluntary_action(exploratory_noise=0.40)
                aa, _, _ = policy.act(obs_e, deterministic=True)
                ah.append(au)
                obs_e, _, _, inf = eval_env.step(au, aa)
                traj_mode.append(eval_env.pos[0].tolist())
                f_mags.append(float(inf["assist_mag"].mean().item()))
                in_targ_final = inf["in_target"]
                dist_final = inf["dist"]

            st = torch.stack(sh[:-k_horizon]).reshape(-1, 4)
            stk = torch.stack(sh[k_horizon:]).reshape(-1, 4)
            at = torch.stack(ah[:-k_horizon]).reshape(-1, 2)
            emp_val = float(eval_q.estimate_mi(st, stk, at, prior_std=0.55).mean().item())

        results[mode_name] = {
            "target_reach_rate": float(in_targ_final.float().mean().item()),
            "final_dist": float(dist_final.mean().item()),
            "user_empowerment_nats": emp_val,
            "assist_force_norm": sum(f_mags) / len(f_mags),
            "final_nu_emp": nu_emp if use_emp_constraint else 0.0,
        }
        sample_trajectories[mode_name] = {"xy": traj_mode}

    return {"metrics": results, "trajectories": sample_trajectories}
