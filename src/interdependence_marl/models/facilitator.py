"""ProToM-Inspired Communication Supporter (Xiao et al., Fig. 7d & Bortoletto et al., 2026).

Monitors belief divergence and cross-modal sensory mismatches between collaborators,
selectively translating and relaying symbolic cues across vocal <-> visual/tactile channels
only when epistemic uncertainty/divergence warrants intervention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ProToMFacilitator:
    """Assistive Communication Supporter bridging orthogonal sensory channels."""

    def __init__(
        self,
        mode: str = "protom",
        jsd_threshold: float = 0.08,
        entropy_threshold: float = 0.55,
    ) -> None:
        assert mode in ("none", "always_on", "protom")
        self.mode = mode
        self.jsd_threshold = jsd_threshold
        self.entropy_threshold = entropy_threshold

    @staticmethod
    def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Compute Jensen-Shannon divergence between distributions p, q of shape `(B, K)`."""
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        m = 0.5 * (p + q)
        kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
        kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm)

    def compute_relay(
        self,
        a_supp: torch.Tensor,
        pred_target_logits: torch.Tensor,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute assistive cross-modal relay `(B, 2, 3)` and binary intervention flag `(B, 2)`.

        Note: Only uses decentralized observable quantities (each agent's own profile in its
        local interface + inferred belief distributions + emitted supportive actions), without
        accessing privileged ground-truth environment target states!
        """
        b = a_supp.shape[0]
        device = a_supp.device
        relay = torch.zeros((b, 2, 3), dtype=torch.float32, device=device)
        intervened = torch.zeros((b, 2), dtype=torch.float32, device=device)

        if self.mode == "none":
            return relay, intervened

        probs = F.softmax(pred_target_logits, dim=-1)  # (B, 2, 3)
        jsd = self._js_divergence(probs[:, 0], probs[:, 1])  # (B,)

        for receiver in range(2):
            sender = 1 - receiver
            s_act = a_supp[:, sender]

            # Extract any symbolic utterance (either vocal 1..3 or visual/text 4..6)
            has_utt = s_act > 0
            sym_from_utt = torch.where(s_act <= 3, s_act - 1, s_act - 4).clamp(0, 2)
            utt_oh = F.one_hot(sym_from_utt, num_classes=3).float() * has_utt.unsqueeze(-1).float()

            # Also allow translating sender's high-confidence belief when sender lacks vocal/visual match
            sender_conf, sender_top = probs[:, sender].max(dim=-1)
            belief_oh = F.one_hot(sender_top, num_classes=3).float() * (
                sender_conf > 0.60
            ).unsqueeze(-1).float()
            candidate_msg = torch.where(has_utt.unsqueeze(-1), utt_oh, belief_oh)
            has_candidate = candidate_msg.sum(dim=-1) > 0.5

            if self.mode == "always_on":
                # Broadcasts continuously whenever any signal/belief exists (high notification load)
                fire = torch.ones(b, dtype=torch.bool, device=device)
                msg = torch.where(
                    has_candidate.unsqueeze(-1),
                    candidate_msg,
                    probs[:, sender],
                )
            else:
                # ProToM selective gating: intervene ONLY when receiver has high belief entropy
                # OR inter-agent belief divergence (JSD) exceeds threshold, AND a cross-modal gap exists
                rec_p = probs[:, receiver].clamp(min=1e-7)
                rec_ent = -(rec_p * rec_p.log()).sum(dim=-1)

                own_phi_rec = obs[:, receiver, 4:8]
                own_phi_snd = obs[:, sender, 4:8]
                vocal_blocked = (own_phi_snd[:, 1] * own_phi_rec[:, 1]) < 0.35
                visual_blocked = (own_phi_snd[:, 0] * own_phi_rec[:, 0]) < 0.35
                modality_gap = vocal_blocked | visual_blocked

                need_grounding = (jsd > self.jsd_threshold) | (rec_ent > self.entropy_threshold)
                fire = has_candidate & need_grounding & modality_gap
                msg = candidate_msg * fire.unsqueeze(-1).float()

            relay[:, receiver] = msg
            intervened[:, receiver] = fire.float()

        return relay, intervened
