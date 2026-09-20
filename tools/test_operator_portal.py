"""Offline checks for validation-only routing and durable attempt reports."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("operator_portal", Path(__file__).with_name("portal.py"))
portal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portal)


class PortalChecks(unittest.TestCase):
    def test_final_routes_fail_before_credentials_or_network(self):
        with patch.object(portal, "api_key", side_effect=AssertionError("must not read credentials")):
            for route in ("survival-simulator/evaluate/queue", "drone-flyby/evaluation",
                          "medical-appointment/validate/queue/id/../../evaluate"):
                with self.subTest(route=route), self.assertRaises(ValueError):
                    portal.req(route, "POST", {})

    def test_environment_key_does_not_require_secret_file(self):
        with patch.dict(os.environ, {"NORDIC_API_KEY": "test-only"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("must not read disk")):
            self.assertEqual(portal.api_key(), "test-only")

    def test_wall_time_excludes_queue(self):
        self.assertEqual(portal.timing({
            "submitted_at": "2026-09-20T09:00:00+00:00",
            "started_at": "2026-09-20T10:00:00+00:00",
            "finished_at": "2026-09-20T10:00:30+00:00",
        }), {"wall_seconds": 30})
        self.assertEqual(portal.timing({}), {})

    def test_validation_queues_once_and_saves_exact_attempt(self):
        attempt = {"score": 1000.0, "errors": [],
                   "started_at": "2026-09-20T10:00:00+00:00",
                   "finished_at": "2026-09-20T10:05:00+00:00"}
        responses = [(200, {"queued_attempt_uuid": "test-id"}),
                     (200, {"status": "in_progress"}),
                     (200, {"status": "done"}), (200, attempt)]
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"NAC_REPORT_DIR": folder}), \
                patch.object(portal, "req", side_effect=responses) as request, \
                patch.object(portal.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            portal.validate("survival-simulator", "http://example.invalid/predict")
            posts = [call for call in request.call_args_list if len(call.args) > 1 and call.args[1] == "POST"]
            self.assertEqual(len(posts), 1)
            report = json.loads((Path(folder) / "survival-simulator-test-id.json").read_text())
            self.assertEqual(report["attempt"], attempt)
            self.assertEqual(report["approx_ms_per_tick"], 30)


if __name__ == "__main__":
    unittest.main()
