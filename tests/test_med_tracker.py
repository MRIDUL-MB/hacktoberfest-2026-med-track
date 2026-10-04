import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

from med_tracker import MedTrackerBot, parse_reminder_times


class MedicationTrackerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.temp_dir.name) / "tracker.json"
        self.tracker = MedTrackerBot(self.data_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_medication_persists_and_sorts_reminder_times(self):
        medication = self.tracker.add_medication(
            "  Vitamin D ",
            "1 tablet",
            "Take with food",
            "21:00, 09:00",
        )

        reloaded = MedTrackerBot(self.data_path)
        self.assertEqual(reloaded.get_medications()[0]["id"], medication["id"])
        self.assertEqual(reloaded.get_medications()[0]["name"], "Vitamin D")
        self.assertEqual(reloaded.get_medications()[0]["reminder_times"], ["09:00", "21:00"])

    def test_day_dose_log_is_persisted_and_updates_checklist(self):
        medication = self.tracker.add_medication(
            "Medicine", "1 tablet", "As directed", "09:00"
        )
        self.tracker.log_dose(medication["id"], date.today().isoformat(), "09:00", "taken")

        reloaded = MedTrackerBot(self.data_path)
        doses = reloaded.get_day_doses()
        self.assertEqual(len(doses), 1)
        self.assertEqual(doses[0]["status"], "taken")

    def test_invalid_time_and_duplicate_time_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_reminder_times("9 AM")
        with self.assertRaises(ValueError):
            parse_reminder_times("09:00, 09:00")

    def test_invalid_dose_status_is_rejected(self):
        medication = self.tracker.add_medication(
            "Medicine", "1 tablet", "As directed", "09:00"
        )
        with self.assertRaises(ValueError):
            self.tracker.log_dose(
                medication["id"], date.today().isoformat(), "09:00", "maybe"
            )

    def test_local_workflow_does_not_require_ai_token(self):
        medication = self.tracker.add_medication(
            "Medicine", "1 tablet", "As directed", ""
        )
        self.assertEqual(medication["name"], "Medicine")
        self.assertEqual(medication["reminder_times"], [])
        self.tracker.api_token = None
        with self.assertRaises(RuntimeError):
            self.tracker.format_instructions_with_ai("Take as directed")

    @patch("med_tracker.requests.post")
    def test_ai_formatter_uses_configured_open_model(self, post):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Take as directed."}}]
        }
        post.return_value = response
        self.tracker.api_token = "test-token"

        result = self.tracker.format_instructions_with_ai("Take as directed")

        self.assertEqual(result, "Take as directed.")
        self.assertEqual(
            post.call_args.args[0],
            "https://router.huggingface.co/v1/chat/completions",
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["model"],
            "HuggingFaceH4/zephyr-7b-beta:featherless-ai",
        )


if __name__ == "__main__":
    unittest.main()
