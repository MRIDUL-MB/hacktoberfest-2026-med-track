# Naman's Local Med Tracker

A small medication-routine helper built for my friend Naman. It keeps a local
medicine list, shows a daily dose checklist, and records doses as taken or
skipped so it's easier to keep track of a routine.

## What it does

- Saves medicine names, label/clinician instructions, and reminder times to a
  JSON file on this computer.
- Shows today's reminder checklist and keeps a taken/skipped history across
  restarts.
- Optionally reformats instructions with the open-weight
  `HuggingFaceH4/zephyr-7b-beta` model.
- Keeps the tracker usable without an API token or an internet connection.

Reminder times must come from Naman's label or healthcare professional. This
app does not calculate doses, send background notifications, or provide medical
advice. It is a checklist, not a substitute for a prescription or care.

## Run it

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the tracker:

```powershell
python med_tracker.py
```

Use **Check today's doses** to see the checklist and **Log a dose** to record an
entry. The local data is stored in `med_tracker_data.json`, next to the script.
That file and `.env` are excluded from Git.

## Optional open-model formatting

The AI feature only runs when explicitly selected. It sends the exact
instructions you paste to Hugging Face's inference router and its selected
provider; do not include personal information. AI wording is only a readability
aid: compare it with the original label, and ask a healthcare professional
about anything unclear.

1. Create a Hugging Face token with Inference Providers permission.
2. Put it in a `.env` file in the project folder:

   ```text
   HF_API_TOKEN=hf_your_token
   ```

3. Choose **Reformat instructions with open AI** in the app.

The model weights are open, but this optional inference call is hosted and needs
internet access and a token. The medicine list and dose history stay local;
they are not sent by the tracker.

## Why open innovation?

The daily checklist works locally without handing a medication history to an
AI service. When a friend wants help making clinician-provided wording easier
to read, an inspectable open-weight model can be used and swapped without
making the local tracker depend on it. The trade-off is explicit: model
formatting requires a network request, while the private core keeps working
offline.

## Tests

Run the local tests without contacting the model service:

```powershell
python -m unittest discover -s tests -v
```
