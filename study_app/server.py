"""Interactive Human-AI Experimental Study Server for Interdependence-MARL.

Serves the accessible HTML5/Web-Speech study interface (`index.html`), runs live
trained PyTorch policies (`ADC-MARL`, `Task-Only MARL`, `Homogeneous-SP`) + `ProToM`
step-by-step alongside a human participant, and logs trial telemetry and post-block
NASA-TLX / Sense-of-Agency ratings to `results/human_study_logs/trials.csv`.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interdependence_marl.envs.adc_posg import AbilityDiversePOSG, EnvConfig  # noqa: E402
from interdependence_marl.models.facilitator import ProToMFacilitator  # noqa: E402
from interdependence_marl.models.networks import InterdependentCTDENet  # noqa: E402
from interdependence_marl.profiles import make_dyad_batch  # noqa: E402


class StudySession:
    """Manages live single-participant human-AI episode state."""

    def __init__(self) -> None:
        self.ckpt_dir = ROOT / "checkpoints"
        self.log_dir = ROOT / "results" / "human_study_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, InterdependentCTDENet] = {}
        self._load_models()
        self.reset_session("ability_sharing", "ADC-MARL", "blv_ambulatory", "full_ability", "protom")

    def _load_models(self) -> None:
        for key, fname in [
            ("ADC-MARL:ability_combining", "adc_marl_combining.pt"),
            ("ADC-MARL:ability_sharing", "adc_marl_sharing.pt"),
            ("Task-Only:ability_sharing", "task_only_sharing.pt"),
        ]:
            path = self.ckpt_dir / fname
            net = InterdependentCTDENet(hidden_dim=96, use_tom=True, use_grounding=True)
            if path.exists():
                net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            net.eval()
            self.models[key] = net

    def reset_session(
        self,
        mode: str,
        condition: str,
        human_profile: str,
        ai_profile: str,
        facilitator_mode: str,
    ) -> dict[str, object]:
        self.mode = mode
        self.condition = condition
        self.human_profile = human_profile
        self.ai_profile = ai_profile
        self.facilitator_mode = facilitator_mode

        model_key = f"{condition}:{mode}"
        if model_key not in self.models:
            model_key = f"ADC-MARL:{mode}"
        self.active_model = self.models[model_key]
        self.facilitator = ProToMFacilitator(mode=facilitator_mode)

        self.env = AbilityDiversePOSG(
            num_envs=1, config=EnvConfig(mode=mode, interdependence_reward=True)
        )
        phi = make_dyad_batch(human_profile, ai_profile, batch_size=1)
        self.obs, self.state = self.env.reset(phi=phi)
        self.h = self.active_model.init_hidden(1)
        self.prev_req = torch.zeros((1, 2), dtype=torch.long)
        self.prev_supp = torch.zeros((1, 2), dtype=torch.long)
        self.done = False
        self.last_info: dict[str, object] = {}
        self.last_ai_supp = 0
        self.last_hat_phi0 = [0.5, 0.5, 0.5, 0.5]
        return self.serialize_state()

    @torch.no_grad()
    def step_human(self, human_req: int, human_supp: int) -> dict[str, object]:
        if self.done:
            return self.serialize_state()

        out = self.active_model.act(
            self.obs, self.state, self.prev_req, self.prev_supp, self.h, deterministic=True
        )
        a_req = out.a_req.clone()
        a_supp = out.a_supp.clone()
        a_req[0, 0] = int(human_req)
        a_supp[0, 0] = int(human_supp)

        self.last_ai_supp = int(a_supp[0, 1].item())
        self.last_hat_phi0 = [round(x, 2) for x in out.pred_partner_phi[0, 1].tolist()]

        relay, fired = self.facilitator.compute_relay(a_supp, out.pred_target_logits, self.obs)
        self.env.set_relay_messages(relay)

        self.obs, self.state, rew, done_t, info = self.env.step(a_req, a_supp)
        self.h = out.h_next
        self.prev_req = a_req
        self.prev_supp = a_supp
        self.done = bool(done_t[0].item())
        self.last_info = {
            "reward": round(float(rew[0].item()), 3),
            "success": bool(info["success"][0].item()),
            "consensual": bool(info["consensual_checkout"][0].item()),
            "override": bool(info["unilateral_override"][0].item()),
            "human_supp_count": int(info["borrower_supp_count"][0].item()),
            "csi": round(float(info["csi"][0].item()), 3),
            "facilitator_fired": bool(fired[0, 0].item() or fired[0, 1].item()),
        }
        return self.serialize_state()

    def serialize_state(self) -> dict[str, object]:
        human_vis = float(self.env.phi[0, 0, 0].item())
        st_pos = self.env.station_pos[0].tolist()
        # Respect sensory masking in UI: hide visual shelf labels when human_vis == 0!
        st_codes = (
            self.env.station_vis_code[0].tolist() if human_vis > 0.2 else ["?", "?", "?"]
        )
        goal_shown = (
            int(self.env.target_code[0].item())
            if (self.mode == "ability_sharing" or human_vis > 0.2)
            else "?"
        )
        return {
            "mode": self.mode,
            "condition": self.condition,
            "human_profile": self.human_profile,
            "ai_profile": self.ai_profile,
            "step": int(self.env.steps[0].item()),
            "max_steps": self.env.cfg.max_steps,
            "human_pos": self.env.pos[0, 0].tolist(),
            "ai_pos": self.env.pos[0, 1].tolist(),
            "stations": [{"pos": p, "vis_code": c} for p, c in zip(st_pos, st_codes)],
            "checkout_pos": [2, 2],
            "goal_code": goal_shown,
            "held_by": int(self.env.held_by[0].item()),
            "ai_last_supp": self.last_ai_supp,
            "ai_inferred_human_phi": self.last_hat_phi0,
            "done": self.done,
            "info": self.last_info,
        }

    def log_survey(self, payload: dict[str, object]) -> None:
        csv_path = self.log_dir / "trials.csv"
        write_header = not csv_path.exists()
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "participant_id": payload.get("participant_id", "anon"),
            "mode": self.mode,
            "condition": self.condition,
            "human_profile": self.human_profile,
            "ai_profile": self.ai_profile,
            "facilitator": self.facilitator_mode,
            "success": self.last_info.get("success", False),
            "override": self.last_info.get("override", False),
            "human_supp_count": self.last_info.get("human_supp_count", 0),
            "csi": self.last_info.get("csi", 0.0),
            "agency_rating": payload.get("agency_rating", ""),
            "tlx_effort": payload.get("tlx_effort", ""),
            "fluency_rating": payload.get("fluency_rating", ""),
        }
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


SESSION = StudySession()
HTML_PATH = Path(__file__).resolve().parent / "index.html"


class StudyHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict[str, object]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            self._send_json(SESSION.serialize_state())
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        if self.path == "/reset":
            st = SESSION.reset_session(
                mode=str(payload.get("mode", "ability_sharing")),
                condition=str(payload.get("condition", "ADC-MARL")),
                human_profile=str(payload.get("human_profile", "blv_ambulatory")),
                ai_profile=str(payload.get("ai_profile", "full_ability")),
                facilitator_mode=str(payload.get("facilitator", "protom")),
            )
            self._send_json(st)
        elif self.path == "/step":
            st = SESSION.step_human(
                int(payload.get("req", 0)),
                int(payload.get("supp", 0)),
            )
            self._send_json(st)
        elif self.path == "/survey":
            SESSION.log_survey(payload)
            self._send_json({"ok": True})
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Human-AI Interdependence Study Server")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), StudyHandler)
    print(f"Human-AI Interdependence Study App listening on http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
