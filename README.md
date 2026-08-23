# Groq Voice Dictation for NVDA

Groq Voice Dictation is an NVDA add-on that provides push-to-toggle voice dictation using Groq Whisper transcription with optional text cleanup.

## Features

- `NVDA+Shift+V` to start and stop dictation
- **Double-press to force-stop.** If the add-on ever gets stuck (mid-transcription, mid-cleanup, or waiting on the confirm window), press `NVDA+Shift+V` twice within half a second to abort it cleanly. The audio is discarded and the in-flight network request is dropped before it can insert anything.
- Optional silence detection for automatic stop
- Groq transcription with optional cleanup modes
- Configurable default microphone
- Typing-first text insertion with paste fallback
- Guided API key setup from the settings panel

## Accuracy and response time

The default recognizer is Whisper Large V3, Groq's recommended model for
error-sensitive transcription. Whisper Large V3 Turbo remains available when
lower cost and slightly lower latency matter more than the lowest word-error
rate. Selecting the spoken language avoids language-detection mistakes.

New configurations start with an empty Whisper prompt because a prompt biases
the recognizer; select one of the focused glossary slots only when its terms
match the current dictation. Raw transcript mode makes one model request. AI
cleanup uses deterministic generation and rejects a result that changes the
opening, pronouns, negation, modality, certainty, or too many lexical words,
falling back to Whisper's raw text instead of guessing missing speech.

The add-on records mono WAV audio and trims leading/trailing silence before upload.
It also opens the Groq connection while you are speaking so connection setup is
normally finished before you press Stop.

## Build

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_addon.ps1
```

The packaged add-on is created in `dist\groqVoiceDictation-0.1.0.nvda-addon`.
