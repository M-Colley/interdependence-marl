"""Ability-Diverse Collaboration in Multi-Agent Reinforcement Learning (ADC-MARL).

Operationalizes Xiao et al.'s (CHI '24) Ability-Diverse Collaboration Framework and
Bennett et al.'s (ASSETS '18) Interdependence Theory as an Ability-Parameterized
Partially Observable Stochastic Game (A-POSG), integrating computational Theory of Mind,
cross-modal grounding, and Ability-Conditioned Unsupervised Partner Design (inspired by
Ruhdorfer, Bortoletto, & Bulling, 2024-2026).
"""

from interdependence_marl.profiles import (
    ABILITY_DIM,
    CANONICAL_ARCHETYPES,
    EVAL_DYADS_COMBINING,
    EVAL_DYADS_SHARING,
    AbilityProfile,
    make_dyad_batch,
)

__all__ = [
    "ABILITY_DIM",
    "CANONICAL_ARCHETYPES",
    "EVAL_DYADS_COMBINING",
    "EVAL_DYADS_SHARING",
    "AbilityProfile",
    "make_dyad_batch",
]
