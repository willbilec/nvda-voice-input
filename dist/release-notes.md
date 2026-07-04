## v0.5.0 — Cleanup model selection fix

### Fixed
- **Cleanup model selection in the settings dialog now saves correctly.** Previously, changing the cleanup model (e.g. switching from a Gemini model to a Groq model) could silently revert because `CleanupDialog._on_model_change` called `self.GetSizer().Layout()` on a `wx.Dialog` that has no sizer of its own, throwing `AttributeError` on every model change. Replaced with `self.Layout()`.

### Why this matters
- Users on Gemini cleanup could not reliably switch to a Groq cleanup model to work around the Gemini free-tier 20 RPD quota limit (HTTP 429 errors with "Quota exceeded for metric: ...free_tier_requests, limit: 20, model: gemini-3.5-flash").

### Notes
- To upgrade: replace your existing add-on installation with the attached groqVoiceDictation-0.5.0.nvda-addon

---

## v0.4.0 — Gemini cleanup client and expanded prompts

### New
- **Gemini API client** (gemini_client.py) — text cleanup via Gemini models
  - Supported models: gemini-2.5-flash-lite, gemini-2.5-flash, gemini-3.5-flash
  - Enter a Gemini API key in Settings → Manage API Keys to enable
- **Expanded cleanup prompts** with detailed ASR mishearing fix rules
  - Concrete fix examples (e.g. \tensor flow → TensorFlow, \type script → TypeScript)
  - False start vs opening-word distinction
  - Paraphrase creep prevention (moderate mode)
  - Hedge, slang, and profanity preservation

### Changed
- Cleanup prompts significantly expanded across all three modes (heavy/moderate/light)
- User message format improved with explicit output constraints
- Model-based routing: cleanup model starting with "gemini" → GeminiClient; otherwise → GroqClient
- Build script and manifest bumped to v0.4.0

### Notes
- **Gemini cleanup is experimental** — may be slow or error on long transcripts (>10 seconds). Use a Groq cleanup model if needed.
- To upgrade: replace your existing add-on installation with the attached groqVoiceDictation-0.4.0.nvda-addon

### Previous (v0.3.0)
- Readback / confirm mode with cancelable preview window
- Fallback microphone improvements
- Silence detection fixes
