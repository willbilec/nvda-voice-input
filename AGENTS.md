# AGENTS.md

## Project: NVDA Voice Input (groqVoiceDictation)

An NVDA add-on for push-to-toggle voice dictation using Groq Whisper transcription with optional AI cleanup via Groq or Gemini.

## Key constraints

- The addon must work within NVDA's Python 3.11+ environment
- Vendored dependencies live in `globalPlugins/groqVoiceDictation/lib/` (requests, urllib3, certifi, charset_normalizer, idna, pyaudio)
- The Groq API key is mandatory (Whisper transcription). The Gemini API key is optional (text cleanup only)
- Build: `powershell -ExecutionPolicy Bypass -File build_addon.ps1`
- Tests: `python -m pytest tests/`

## Known issues

- **Gemini cleanup is experimental / work in progress.** The expanded cleanup prompts improve quality but may be slow or error on long transcripts (>10 seconds of speech). If you encounter issues with Gemini, switch to a Groq cleanup model via the Settings dialog.
- The fallback microphone feature auto-switches if the primary mic produces silence during the preflight period

## Architecture

- `__init__.py` — GlobalPlugin entry point, orchestrates the recording → transcription → cleanup → insertion pipeline
- `audio_recorder.py` — PyAudio recording with silence detection
- `groq_client.py` — Groq API: Whisper transcription + chat-based cleanup (also exports `build_cleanup_messages`, `strip_thinking_tags`)
- `gemini_client.py` — Gemini API: cleanup-only client with independent prompt system
- `config_manager.py` — NVDA config spec and constants
- `settings_panel.py` — NVDA settings UI dialogs
- `text_inserter.py` — SendInput-based text insertion with paste fallback

## Gemini cleanup notes

- Model routing: `cleanupModel` starts with "gemini" → GeminiClient; otherwise → GroqClient
- Prompts are independent from Groq's — `_gemini_cleanup_system_prompt()` in gemini_client.py
- Both clients share `strip_thinking_tags()` from groq_client.py for post-processing
- Available models: gemini-2.5-flash-lite, gemini-2.5-flash, gemini-3.5-flash
