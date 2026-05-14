# Groq Voice Dictation for NVDA

Groq Voice Dictation is an NVDA add-on that provides push-to-toggle voice dictation using Groq Whisper transcription with optional text cleanup.

## Features

- `NVDA+Shift+V` to start and stop dictation
- Optional silence detection for automatic stop
- Groq transcription with optional cleanup modes
- Configurable default microphone
- Typing-first text insertion with paste fallback
- Guided API key setup from the settings panel

## Build

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_addon.ps1
```

The packaged add-on is created in `dist\groqVoiceDictation-0.1.0.nvda-addon`.
