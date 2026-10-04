import json
import os
import uuid
from datetime import date, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv


def parse_reminder_times(value):
    times = []
    for part in value.split(","):
        time_text = part.strip()
        if not time_text:
            continue
        try:
            parsed = datetime.strptime(time_text, "%H:%M")
        except ValueError as exc:
            raise ValueError("Enter reminder times in 24-hour HH:MM format.") from exc
        if parsed.strftime("%H:%M") != time_text:
            raise ValueError("Enter reminder times in 24-hour HH:MM format.")
        if time_text in times:
            raise ValueError(f"Reminder time {time_text} was entered more than once.")
        times.append(time_text)
    return sorted(times)


class MedTrackerBot:
    def __init__(self, data_path=None):
        load_dotenv()
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.api_token = os.getenv("HF_API_TOKEN")
        self.data_path = Path(data_path or Path(__file__).with_name("med_tracker_data.json"))
        self.data = self._load_data()

    def _load_data(self):
        if not self.data_path.exists():
            return {"medications": [], "dose_log": []}

        try:
            with self.data_path.open(encoding="utf-8") as data_file:
                data = json.load(data_file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{self.data_path} contains invalid JSON. Back it up before repairing it."
            ) from exc

        if (
            not isinstance(data, dict)
            or not isinstance(data.get("medications"), list)
            or not isinstance(data.get("dose_log"), list)
        ):
            raise ValueError(f"{self.data_path} has an unsupported medication data format.")
        return data

    def _save_data(self):
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.data_path.with_suffix(self.data_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as data_file:
            json.dump(self.data, data_file, indent=2, ensure_ascii=False)
        temporary_path.replace(self.data_path)

    def add_medication(self, name, dosage, instructions, reminder_times):
        name = name.strip()
        dosage = dosage.strip()
        instructions = instructions.strip()
        if not name or not dosage or not instructions:
            raise ValueError("Medicine name, dose, and instructions are required.")

        medication = {
            "id": str(uuid.uuid4()),
            "name": name,
            "dosage": dosage,
            "instructions": instructions,
            "reminder_times": parse_reminder_times(reminder_times),
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.data["medications"].append(medication)
        self._save_data()
        return medication

    def get_medications(self):
        return list(self.data["medications"])

    def get_day_doses(self, day=None):
        day_text = (day or date.today()).isoformat() if isinstance(day, date) or day is None else day
        latest_status = {}
        for entry in self.data["dose_log"]:
            if entry.get("date") == day_text:
                latest_status[(entry.get("medication_id"), entry.get("time"))] = entry.get("status")

        doses = []
        for medication in self.data["medications"]:
            for reminder_time in medication["reminder_times"]:
                doses.append(
                    {
                        "medication_id": medication["id"],
                        "name": medication["name"],
                        "dosage": medication["dosage"],
                        "time": reminder_time,
                        "status": latest_status.get((medication["id"], reminder_time), "pending"),
                    }
                )
        return sorted(doses, key=lambda dose: (dose["time"], dose["name"].casefold()))

    def log_dose(self, medication_id, day, reminder_time, status):
        if status not in {"taken", "skipped"}:
            raise ValueError("Dose status must be 'taken' or 'skipped'.")
        try:
            date.fromisoformat(day)
            parsed_time = datetime.strptime(reminder_time, "%H:%M")
            if parsed_time.strftime("%H:%M") != reminder_time:
                raise ValueError("Reminder time must use HH:MM format.")
        except ValueError as exc:
            raise ValueError("Use a valid date and a reminder time in HH:MM format.") from exc

        medication = next(
            (med for med in self.data["medications"] if med["id"] == medication_id),
            None,
        )
        if medication is None or reminder_time not in medication["reminder_times"]:
            raise ValueError("That medication reminder could not be found.")

        self.data["dose_log"].append(
            {
                "medication_id": medication_id,
                "date": day,
                "time": reminder_time,
                "status": status,
                "logged_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._save_data()

    def format_instructions_with_ai(self, instructions):
        if not self.api_token:
            raise RuntimeError(
                "AI formatting needs HF_API_TOKEN in your .env file. "
                "You can still use the local tracker without it."
            )

        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_token}"},
            json={
                "model": "HuggingFaceH4/zephyr-7b-beta:featherless-ai",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user's medication instructions as a clearer, concise "
                            "format. Do not add, infer, or change doses, times, or medical "
                            "advice. If the instructions are ambiguous, say to check with a "
                            "healthcare professional."
                        ),
                    },
                    {"role": "user", "content": instructions},
                ],
                "max_tokens": 100,
            },
            timeout=20,
        )
        response.raise_for_status()
        try:
            content = response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise RuntimeError("The AI service returned an unexpected response.") from exc
        if not content:
            raise RuntimeError("The AI service returned an empty response.")
        return content

    def _add_medication_from_prompt(self):
        print("\n--- Add a Medication ---")
        name = input("Medicine name: ").strip()
        dosage = input("Dose as written on the label (for example, 1 tablet): ").strip()
        instructions = input("Instructions from the label or clinician: ").strip()
        print("Reminder times are for your reminders only; enter times confirmed by Naman.")
        reminder_times = input("Reminder times, 24-hour HH:MM separated by commas: ").strip()
        try:
            medication = self.add_medication(name, dosage, instructions, reminder_times)
        except ValueError as exc:
            print(f"Could not add medication: {exc}")
            return

        print(f"\nSaved {medication['name']} on this device.")
        if not medication["reminder_times"]:
            print("No reminder times set. You can still use the tracker without reminders.")

    def _show_medications(self):
        print("\n--- Medicines ---")
        medications = self.get_medications()
        if not medications:
            print("No medicines added yet.")
            return
        for medication in medications:
            reminders = ", ".join(medication["reminder_times"]) or "none set"
            print(f"\n{medication['name']} — {medication['dosage']}")
            print(f"  Instructions: {medication['instructions']}")
            print(f"  Reminder times: {reminders}")

    def _show_today(self):
        print(f"\n--- Today's Dose Checklist ({date.today().isoformat()}) ---")
        doses = self.get_day_doses()
        if not doses:
            if not self.get_medications():
                print("No medicines added yet.")
            else:
                print("No reminder times set. Add times confirmed by Naman's clinician.")
            return
        for dose in doses:
            print(
                f"[{dose['status'].upper():7}] {dose['time']} — "
                f"{dose['name']} ({dose['dosage']})"
            )

    def _log_today_dose(self):
        doses = [
            dose
            for dose in self.get_day_doses()
            if dose["status"] == "pending"
        ]
        if not doses:
            print("No unlogged doses with reminder times today.")
            return

        print("\n--- Log a Dose ---")
        for index, dose in enumerate(doses, start=1):
            print(f"{index}. {dose['time']} — {dose['name']} ({dose['dosage']})")
        choice = input("Select a dose, or press Enter to cancel: ").strip()
        if not choice:
            return
        try:
            selection = int(choice)
            if selection < 1:
                raise ValueError
            dose = doses[selection - 1]
        except (ValueError, IndexError):
            print("That dose selection is not valid.")
            return

        status = input("Enter 't' for taken or 's' for skipped: ").strip().lower()
        if status not in {"t", "s"}:
            print("Dose not changed; enter 't' or 's'.")
            return
        self.log_dose(
            dose["medication_id"],
            date.today().isoformat(),
            dose["time"],
            "taken" if status == "t" else "skipped",
        )
        print(f"Logged as {'taken' if status == 't' else 'skipped'}.")

    def _format_instructions(self):
        if not self.api_token:
            print("AI formatting is unavailable: set HF_API_TOKEN in .env first.")
            return
        print(
            "This sends only the instruction text you enter to Hugging Face "
            "and its inference provider. Do not include personal details."
        )
        if input("Continue? (y/N): ").strip().lower() != "y":
            print("Nothing was sent.")
            return
        instructions = input("Paste the exact label/clinician instructions: ").strip()
        if not instructions:
            print("No instructions entered.")
            return
        try:
            print("\nAI wording (verify against the original instructions):")
            print(self.format_instructions_with_ai(instructions))
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            print(f"AI formatting failed: {exc}")

    def run(self):
        print("Naman's Local Med Tracker")
        print("Private local tracking; this app does not provide medical advice.")
        while True:
            print("\n1. Add a medicine")
            print("2. View medicines")
            print("3. Check today's doses")
            print("4. Log a dose")
            print("5. Reformat instructions with open AI (optional)")
            print("6. Exit")
            choice = input("Choose an option (1-6): ").strip()
            if choice == "1":
                self._add_medication_from_prompt()
            elif choice == "2":
                self._show_medications()
            elif choice == "3":
                self._show_today()
            elif choice == "4":
                self._log_today_dose()
            elif choice == "5":
                self._format_instructions()
            elif choice == "6":
                print("Take care, Naman!")
                return
            else:
                print("Please choose a number from 1 to 6.")


if __name__ == "__main__":
    MedTrackerBot().run()
