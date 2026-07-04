# Changelog

## v0.4.0 (2026-07-03)

### Added
- **Gemini API client** for text cleanup (`gemini_client.py`) — supports gemini-2.5-flash-lite, gemini-2.5-flash, gemini-3.5-flash
- Gemini API key management in the Settings → Manage API Keys dialog
- Gemini models appear in the cleanup model dropdown when a Gemini key is set
- **Expanded cleanup prompts** for both Groq and Gemini clients with detailed rules for:
  - Fixing ASR mishearings with concrete examples (e.g. `tensor flow` → `TensorFlow`)
  - Distinguishing false starts from sentence-opening words
  - Preserving hedges, slang, profanity, discourse markers
  - Paraphrase creep prevention in moderate mode

### Changed
- Cleanup prompts significantly expanded across all three modes (heavy/moderate/light)
- User message format improved to explicitly forbid thinking tags, explanations, and code fences
- Model-based routing: Gemini client used when `cleanupModel` starts with "gemini"; otherwise Groq

### Known issues
- Gemini cleanup is experimental — may be slow or error on long transcripts (>10 seconds)

---

## v0.3.0 (2026-07-03)

### Added
- **Readback / confirm mode**: speak raw transcript, speak after insertion, or confirm before insertion
- Confirm window: displays transcribed text, auto-inserts after configurable timeout, cancelable with Escape
- `readbackMode` config setting (off / after / confirm)
- `confirmTimeout` config setting (2–15 seconds)
- `speakRawTranscript` debug setting to hear unprocessed transcripts

### Changed
- `_process_recording` pipeline expanded with readback/confirm branching
- Space and Escape gestures bound during confirm window for accept/cancel

---

## v0.2.0 (2026-07-02)

### Added
- Fallback microphone support with automatic silence detection
- Microphone level sampler for preflight silence check
- Silence detection improvements

---

## v0.1.0 (2026-07-01)

### Added
- Initial public release
- Push-to-toggle voice dictation via NVDA+Shift+V
- Groq Whisper transcription (whisper-large-v3-turbo)
- Groq chat-based text cleanup (heavy/moderate/light/raw modes)
- PyAudio recording with silence-based auto-stop
- SendInput text insertion with paste fallback
- NVDA settings UI for all configuration
