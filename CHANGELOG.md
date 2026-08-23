# Changelog

## Unreleased

### Accuracy
- **Cleanup can no longer silently change the speaker's intent.** Groq and Gemini cleanup now run at temperature 0 and pass through a deterministic fidelity gate. If cleanup changes the opening word, grammatical person or pronouns, negation, modal force (`can`, `should`, `must`, and similar words), certainty, conditionals, or too many lexical words, the add-on rejects it and inserts the raw Whisper transcript. Short utterances under eight words allow only capitalization and punctuation changes. This directly prevents `Do some research` from becoming `I should do some research`.
- **Spoken words are no longer removed on a single suspicious diagnostic.** Segment filtering now follows Whisper's documented conjunction: a segment is treated as silence only when `no_speech_prob` is high and `avg_logprob` is low. High compression ratio is logged rather than used to erase speech, because Groq has already performed Whisper's decode fallback. A physically spoken phrase such as `thank you` is no longer discarded merely because it appears on a generic hallucination list.
- **Accuracy-first transcription defaults.** New configurations default to full `whisper-large-v3`, which Groq recommends for error-sensitive work, while Turbo remains available. The default active prompt slot is empty so an unrelated glossary cannot bias general dictation; focused vocabulary prompts remain available when relevant.
- Automatic prompt-free retry now skips the second request when the active prompt is already empty, because it would be an identical temperature-0 transcription rather than independent evidence.

### Performance
- **Network setup is hidden behind recording.** Starting dictation now pre-warms Groq's authenticated HTTPS connection on a background thread. Groq clients share a thread-safe connection pool across consecutive dictations, and a Groq cleanup request reuses the transcription client's session. This removes repeat DNS/TCP/TLS setup from the stop-to-insert critical path while keeping force-cancel workers isolated.
- **The default fast path makes one model request.** New/default configurations use raw Whisper output (which already includes punctuation) and leave suspicious-result auto-retry off. AI cleanup and retry remain opt-in for users who prefer the quality tradeoff. Existing explicitly saved settings are preserved.
- **Cleanup prompts are 74-79% shorter.** Light/moderate/heavy system prompts fell from 449/942/506 words to 112/196/129 words while retaining opening-word, meaning, short-utterance, hallucination, and prompt-injection safeguards. This reduces request bytes and time to first token. Qwen 3.6 cleanup explicitly uses its non-thinking mode; GPT-OSS retains low reasoning effort.
- **Stopping and insertion do less foreground work.** Microphone callbacks cache whether speech occurred, avoiding an entire-recording rescan when Stop is pressed. Unicode insertion uses 256-character batches with no fixed inter-batch sleeps, and paste fallback no longer waits 50 ms after NVDA's synchronous clipboard write.
- **Failure response is faster.** Groq now uses separate connect/read timeouts, failing connection establishment after 5 seconds instead of leaving every failure mode under a single 60-second timeout. Client and server request timings are logged for future latency analysis.

### Changed
- Added `qwen/qwen3.6-27b` to cleanup choices and removed `qwen/qwen3-32b` and `meta-llama/llama-4-scout-17b-16e-instruct`, which Groq will shut down on July 17, 2026.

### Fixed
- **Last dictation stayed on the clipboard after a paste-fallback insertion.** `TextInserter._paste_text` took the "backup" of the user's clipboard by calling `api.getClipData()` *after* `api.copyToClip(text)` had already overwritten the clipboard. The "backup" was therefore the dictation text itself, and the deferred `_restore_clipboard` put the dictation text back on the clipboard after the paste, instead of restoring the user's original content. Fixed by taking the backup BEFORE the copy. Symptom: a user who had typed or copied something to the clipboard, then used the add-on (which fell back to paste), would find the dictation text sitting on the clipboard indefinitely — appearing to come from nowhere if paste fallback had since been turned off in Debugging settings.

### Tests
- Added cleanup-fidelity regression coverage for command/statement changes, pronouns, negation, modals, short utterances, contraction expansion, proper-noun repair, Groq/Gemini raw fallback, conservative segment filtering, deterministic cleanup temperature, accuracy-first defaults, and no-op retry suppression.
- New `tests/test_text_inserter.py` with 13 unit tests pinning the clipboard-order contract (`getClipData` must be called before `copyToClip`; the restore must receive the user's original clipboard content, not the dictation text) and the `insert()` routing (typing-first with paste as fallback, paste disabled in console when fallback is off, clipboard untouched when paste fallback is disabled and typing succeeds or fails).

## v0.7.1 (2026-07-06)

### Changed
- **Cleanup is roughly twice as fast for default users.** The default cleanup model is `openai/gpt-oss-20b`, a reasoning model that does internal chain-of-thought before producing the cleaned text. The old code let it reason at its default (medium) effort, which is wasted compute for a fully rule-bound cleanup prompt with explicit gates. The Groq client now sends `reasoning_effort: "low"` for gpt-oss models (and only for gpt-oss — other models silently ignore the parameter). The parameter is a new config key, `cleanupReasoningEffort`, with default `"low"` and a three-position `Low (fastest) / Medium (balanced) / High (most thorough)` dropdown in the Cleanup settings dialog. The dropdown is hidden for non-gpt-oss cleanup models because they don't honor the parameter; the user shouldn't see a control that does nothing. Asymmetric fallback: if a hard case appears, bump the dropdown to medium or high. Existing users pick up the speedup automatically on the next launch because the default is now "low".
- **Hard `max_completion_tokens` cap on every cleanup call.** Without a cap, a reasoning model can keep generating thinking tokens indefinitely. The Groq client now sends `max_completion_tokens: 2000` and the Gemini client sends `maxOutputTokens: 2000`. 2000 is comfortably more than any realistic transcript plus reasoning. Safety + speed: a runaway cleanup can no longer burn 30s of generation.
- **`include_reasoning: false` on the Groq cleanup call.** The reasoning field is never read by the add-on, so asking Groq to return the bytes is wasted wire. Excluded by default.
- **Prompt-cache hit rate is now logged.** Groq prompt caching (Aug 2025) is automatic and works on our cleanup call because the system prompt is identical across calls and is always the first message — exact prefix match. The first call is cold; the second+ should be a near-100% cache hit. The new log line, `Groq cleanup cache: N/M prompt tokens cached (P.P%)`, makes this visible. If the cache hit rate ever drops, we'll see it in the NVDA log without needing a debug session.

### Tests
- New `CleanupPostBodyTests` class with 9 tests pinning the wire shape: gpt-oss cleanup uses `reasoning_effort: "low"` by default, accepts custom values, and is honored by both `gpt-oss-20b` and `gpt-oss-120b`; non-gpt-oss models (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `qwen/qwen3-32b`, `meta-llama/llama-4-scout`) do NOT get the parameter; `max_completion_tokens` cap is present; `include_reasoning: false` is present; `raw` mode does not hit the network; cache hit rate is logged; and the message order (system first, user second) is preserved so the prompt cache prefix matches.
- New `GeminiCleanupPostBodyTests` class with 2 tests pinning the Gemini wire shape: `maxOutputTokens` cap is present and temperature stays at 0.3.
- New `test_cleanup_reasoning_effort_in_confspec` in the existing conftest block, pinning the new config key, its type, and its default.

## v0.7.0 (2026-07-05)

### Changed
- **Moderate cleanup now fixes obvious ASR mishearings.** The previous moderate prompt was so restrictive (10 "do NOT" bullets vs 6 "do" bullets) that the model had nothing actionable to fix on Whisper's already-clean output — `we` stayed `we` even when the speaker was addressing a single listener, `tensor flow` stayed `tensor flow` instead of `TensorFlow`, homophone swaps (`their/there/they're`, `your/you're`, `its/it's`, `to/too/two`) stayed wrong. The new moderate prompt grants a narrow ASR-mishearing license: the model may **replace** a word (it may not add or remove) ONLY when the surrounding context makes the original clearly wrong. Four categories are licensed with concrete examples: (1) pronoun mishears (`we` -> `you` when addressing a single listener, `I` -> `you` when giving instructions, `he` <-> `she` when a name makes the original impossible), (2) homophones that change meaning, (3) compound terms and proper nouns the recogniser split (`tensor flow` -> `TensorFlow`, `API gate way` -> `API gateway`, `post gress SQL` -> `PostgreSQL`), and (4) contraction expansions the speaker clearly intended (`wanna` -> `want to`). Every fix must pass the "human transcriber" test: would a transcriber, listening to the audio, change exactly this word? Asymmetric risk rule: false-positive fixes (over-correcting) are worse than false-negatives (leaving a mishear in). The license is suspended for short utterances (under 8 words) and capped at 5-10% of the transcript length to prevent paraphrase creep. All other conservative rules (no opening-word drops, no hedge removal, no slang sanitization, no pronoun changes for "clarity") remain in force. Mirrored in the Gemini prompt so both providers behave identically.

### Tests
- 6 new tests in `tests/test_groq_client.py` pinning the new moderate prompt: ASR license is present, the three specific categories (pronouns, homophones, compound terms) are listed, the concrete `TensorFlow` / `API gateway` / `PostgreSQL` examples are included, the human-transcriber test and 5-10% paraphrase-creep cap are in the prompt, and short utterances (under 8 words) are protected from ASR fixes. Updated `test_moderate_forbids_pronoun_changes_for_clarity` to assert the new gated rule (pronoun changes allowed only as ASR mishears, not for "clarity" or "consistency") and removed the obsolete `test_moderate_does_not_fix_asr_mishearings` / `test_moderate_forbids_replacement` tests that pinned the old over-conservative behavior.
- New `GeminiModeratePromptParityTests` class with 5 tests asserting the Gemini moderate prompt matches the Groq one on the ASR license, examples, no-rephrasing, short-utterance protection, and the 5-10% cap. This pins provider parity so future prompt edits to one side surface as a test failure on the other.

## v0.6.1 (2026-07-05)

### Fixed
- **Custom microphone silently reverting to "System default" in the settings dialog.** `list_input_devices()` filters out devices on host APIs it does not trust (MME, "Microsoft Sound Mapper", generic "Input") and any device that fails to enumerate on a particular run. When the user's saved microphone was on a filtered host API, `MicrophoneDialog._index_for_device()` returned `0` for the missing entry — silently selecting "System default microphone" in the dialog. Clicking OK then overwrote the user's saved device index with `-1`. Added `ensure_device_in_choices()` in `settings_panel.py` and call it from `makeSettings()` so the user's saved device (primary and fallback) is always injected as an explicit choice list entry, with the label "Saved microphone (device N, not currently detected)" when the device is not currently present. The dialog now shows the saved device as the active selection, and clicking OK preserves it.
- **Add-on stuck on "Processing" after a PortAudio stop_stream() hang.** Some Windows audio drivers (notably Realtek and several USB devices) raise `OSError [Errno -9987] Wait timed out` from PortAudio's `stop_stream()` call. The old `AudioRecorder.stop()` let the exception propagate, and `_stop_and_process` only caught `AudioRecorderError`, so the `OSError` escaped the script handler entirely. The result: `_processing = True` with no worker thread to ever reset it, leaving the add-on permanently stuck. `AudioRecorder.stop()` now isolates `stop_stream()` and `close()` so a PortAudio hang cannot prevent `close()` from running or the caller from getting a wav path back. `_stop_and_process` also catches any other unexpected exception during `recorder.stop()` and releases the `_processing` flag with a "Could not stop the recorder." notification. This is the root cause of the "Stuck on processing" symptom the double-press escape hatch was a workaround for.

### Tests
- New `tests/test_microphone_preserve.py` with 11 unit tests covering `ensure_device_in_choices()` (appends missing devices, skips duplicates, skips the `-1` sentinel, preserves order, default label format, the regression where `_index_for_device` would have returned `0`), `AudioRecorder.stop()` (OSError on `stop_stream` no longer propagates, state is cleared, second call raises), and `_stop_and_process` (OSError during stop no longer leaves `_processing` set).

## v0.6.0 (2026-07-05)

### Added
- **Double-press escape hatch.** Pressing the toggle shortcut (NVDA+Shift+V) twice within 500 ms now force-stops the add-on no matter what state it is in: idle, recording, mid-transcription, mid-cleanup, or sitting in the readback/confirm window. The second press tears down the active recorder, cancels the preflight and confirm timers, clears the pending-text buffer, and bumps a monotonic cancel token. The processing worker (if any) checks that token at every gate — before transcribing, after transcribing, after cleanup, and right before inserting text — and bails out instead of dumping audio into the user's document. This is the user-facing fix for the "stuck on processing" symptom: when the add-on is wedged, two quick taps of the shortcut unstick it without restarting NVDA.

### Changed
- `_process_recording` now checks the cancel token at four checkpoints (start, post-transcribe, post-cleanup, pre-insert) and bails early when it has been bumped. The `finally` clause also guards against clobbering a fresh dictation's `_processing` flag if a stale worker finishes after a force-cancel.
- `_execute_pending_insert` guards against `_pending_text` being cleared by a force-cancel between the confirm timer firing and the callback running (both happen on the main thread, so the race is rare, but cheap to defend against).
- `script_toggleVoiceDictation` runs the double-press check before the "still processing" guard so the escape hatch works while the add-on is busy.

### Tests
- New `tests/test_force_cancel.py` with 22 unit tests covering: single-press normal toggle, in-window double-press detection, out-of-window second press, processing-state single-press block, processing-state double-press escape, recording-in-progress double-press abort, force-cancel state teardown (token bump, processing clear, pending-text clear, recorder stop + wav delete, recorder-stop exception handling, preflight cancel, confirm-gesture clear, safe-when-idle), cancel-token stale-vs-current discrimination, and confirm-window race hardening.

## v0.5.0 (2026-07-04)

### Fixed
- **Cleanup model selection not sticking in settings dialog.** `CleanupDialog._on_model_change` called `self.GetSizer().Layout()`, but `CleanupDialog` is a `wx.Dialog` (not a `SettingsPanel`) and has no sizer of its own, so `GetSizer()` returned `None` and `.Layout()` threw `AttributeError: 'NoneType' object has no attribute 'Layout'` on every model change. The exception left the dialog's event loop in a confused state, causing subsequent OK clicks to intermittently fail to persist the new value to `nvda.ini`. Changed to `self.Layout()` with a comment explaining why. Symptom: users trying to switch the cleanup model from Gemini to Groq (or vice versa) sometimes found the change reverted after restarting NVDA.

### Added
- Three regression tests in `CleanupDialogRegressionTests` that assert the buggy `self.GetSizer().Layout()` cannot be reintroduced and that the rationale comment is preserved.

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
