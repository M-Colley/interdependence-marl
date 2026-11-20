"""Continuous ability profiles and canonical ability-diverse dyads.

Follows Ability-Based Design (Wobbrock et al., 2011), Conceptual User Modeling
(Nolte et al., 2022), and Xiao et al.'s (CHI '24) Ability-Diverse Collaboration
Framework by representing agents via continuous multi-dimensional capability vectors
rather than binary deficit labels:
    phi = (vis, aud, reach, motor) in [0, 1]^4
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

ABILITY_DIM = 4
ABILITY_NAMES = ("vis", "aud", "reach", "motor")


@dataclass(frozen=True)
class AbilityProfile:
    """Continuous 4D capability profile for an agent in [0, 1]^4."""

    name: str
    vis: float    # Visual acuity / global visual perception & gesture channel
    aud: float    # Auditory/tactile perception & vocal speech channel
    reach: float  # Upper-reach / heavy dispenser actuation capability
    motor: float  # Motor execution precision (1 - tremor/overshoot slip)

    def to_tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.tensor(
            [self.vis, self.aud, self.reach, self.motor],
            dtype=torch.float32,
            device=device,
        )


CANONICAL_ARCHETYPES: dict[str, AbilityProfile] = {
    "full_ability": AbilityProfile("Full-Ability", vis=1.0, aud=1.0, reach=1.0, motor=1.0),
    "blv_ambulatory": AbilityProfile("BLV-Ambulatory", vis=0.0, aud=1.0, reach=1.0, motor=1.0),
    "wheelchair_sighted": AbilityProfile(
        "Wheelchair-Sighted", vis=1.0, aud=1.0, reach=0.0, motor=0.95
    ),
    "dhh_sighted": AbilityProfile("DHH-Sighted", vis=1.0, aud=0.0, reach=1.0, motor=1.0),
    "low_vision_tremor": AbilityProfile(
        "LowVision-Tremor", vis=0.30, aud=1.0, reach=0.85, motor=0.55
    ),
    "wheelchair_dhh": AbilityProfile(
        "Wheelchair-DHH", vis=1.0, aud=0.0, reach=0.0, motor=0.90
    ),
    "motor_ataxia": AbilityProfile("Motor-Ataxia", vis=1.0, aud=1.0, reach=0.25, motor=0.45),
}

# Canonical evaluation dyads for Ability Combining (Xiao et al., Sec. 4.2.2 & 5.1.1)
EVAL_DYADS_COMBINING: list[tuple[str, str, str]] = [
    ("Symmetric Full-Ability", "full_ability", "full_ability"),
    # Bennett et al. (2018) vending machine case study: wheelchair user + blind partner
    ("Wheelchair + BLV (Vending)", "wheelchair_sighted", "blv_ambulatory"),
    # Held-out continuous asymmetry: wheelchair DHH + full-hearing/reach BLV-like or full partner
    ("Wheelchair-DHH + Full", "wheelchair_dhh", "full_ability"),
    # Held-out neurodegenerative motor ataxia + BLV partner
    ("Motor-Ataxia + BLV", "motor_ataxia", "blv_ambulatory"),
    # Held-out partial low-vision with tremor + wheelchair user
    ("LowVision-Tremor + Wheelchair", "low_vision_tremor", "wheelchair_sighted"),
    # Xiao et al. Sec. 5.1.1 orthogonal sensory channels: Blind-Hearing + Deaf-Sighted (Wheelchair)
    ("BLV + Wheelchair-DHH (Cross-Modal)", "blv_ambulatory", "wheelchair_dhh"),
]

# Canonical evaluation dyads for Ability Sharing (Borrower = Agent 0, Provider = Agent 1)
EVAL_DYADS_SHARING: list[tuple[str, str, str]] = [
    ("BLV Borrower + Sighted Provider", "blv_ambulatory", "full_ability"),
    ("Wheelchair Borrower + Ambulatory Provider", "wheelchair_sighted", "full_ability"),
    ("Wheelchair-DHH Borrower + Sighted Provider", "wheelchair_dhh", "full_ability"),
    ("LowVision-Tremor Borrower + Sighted Provider", "low_vision_tremor", "full_ability"),
    # PwD-to-PwD Ability Sharing (Xiao et al. Sec. 4.2.3): Wheelchair Borrower + BLV Provider
    ("Wheelchair Borrower + BLV Provider (PwD->PwD)", "wheelchair_sighted", "blv_ambulatory"),
    # Cross-modal chasm (Xiao et al. Sec. 5.3.2): BLV Borrower + DHH-Sighted Provider
    ("BLV Borrower + DHH Provider (Cross-Modal)", "blv_ambulatory", "dhh_sighted"),
]


def make_dyad_batch(
    key_0: str,
    key_1: str,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a (batch_size, 2, 4) tensor of capability profiles for a named dyad."""
    p0 = CANONICAL_ARCHETYPES[key_0].to_tensor(device=device)
    p1 = CANONICAL_ARCHETYPES[key_1].to_tensor(device=device)
    dyad = torch.stack([p0, p1], dim=0).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    return dyad
