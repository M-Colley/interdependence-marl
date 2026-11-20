"""Primal-Dual Lagrangian Optimization for Constrained Ability-Parameterized POSGs (CA-POSGs).

Solves the constrained interdependence problem without manual penalty tuning:
    max_theta  J_task(pi_theta)
    s.t.       C_override(pi_theta) <= tau_override
               C_burden(pi_theta)   <= tau_burden
               E_borrower(pi_theta) >= tau_empowerment
via projected dual gradient ascent on Lagrange multipliers (nu_override, nu_burden, nu_emp).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LagrangianConfig:
    tau_override: float = 0.05       # Max 5% unilateral paternalistic overrides
    tau_burden: float = 5.00         # Max ~5.0 supportive utterances per episode
    tau_empowerment: float = 0.35    # Min borrower control/agency index
    lr_override: float = 0.025
    lr_burden: float = 0.0015
    lr_emp: float = 0.025
    max_nu_override: float = 0.85
    max_nu_burden: float = 0.028
    max_nu_emp: float = 0.65
    ema_decay: float = 0.82
    warmup_progress: float = 0.38    # Allow task exploration before tightening duals


@dataclass
class PrimalDualController:
    """Maintains EMA constraint estimates and updates dual multipliers online."""

    cfg: LagrangianConfig = field(default_factory=LagrangianConfig)
    nu_override: float = 0.05
    nu_burden: float = 0.001
    nu_emp: float = 0.05

    ema_override: float = 0.50
    ema_burden: float = 8.00
    ema_emp: float = 0.50
    initialized: bool = False

    def step_duals(
        self,
        progress: float,
        batch_override_rate: float,
        batch_burden: float,
        batch_empowerment: float,
    ) -> dict[str, float]:
        """Update EMA constraint estimates and take a projected dual ascent step."""
        d = self.cfg.ema_decay
        if not self.initialized:
            self.ema_override = batch_override_rate
            self.ema_burden = batch_burden
            self.ema_emp = batch_empowerment
            self.initialized = True
        else:
            self.ema_override = d * self.ema_override + (1.0 - d) * batch_override_rate
            self.ema_burden = d * self.ema_burden + (1.0 - d) * batch_burden
            self.ema_emp = d * self.ema_emp + (1.0 - d) * batch_empowerment

        if progress >= self.cfg.warmup_progress:
            viol_over = self.ema_override - self.cfg.tau_override
            self.nu_override = min(
                self.cfg.max_nu_override,
                max(0.0, self.nu_override + self.cfg.lr_override * viol_over),
            )

            viol_burd = self.ema_burden - self.cfg.tau_burden
            self.nu_burden = min(
                self.cfg.max_nu_burden,
                max(0.0, self.nu_burden + self.cfg.lr_burden * viol_burd),
            )

            viol_emp = self.cfg.tau_empowerment - self.ema_emp
            self.nu_emp = min(
                self.cfg.max_nu_emp,
                max(0.0, self.nu_emp + self.cfg.lr_emp * viol_emp),
            )

        return {
            "nu_override": self.nu_override,
            "nu_burden": self.nu_burden,
            "nu_emp": self.nu_emp,
            "ema_override": self.ema_override,
            "ema_burden": self.ema_burden,
            "ema_emp": self.ema_emp,
        }
