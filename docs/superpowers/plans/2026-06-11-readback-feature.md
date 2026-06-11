# Read-Back Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional NVDA speech read-back of the final dictated text, either after insertion or before insertion with a cancel window.

**Architecture:** Three settings-driven paths in `_process_recording`: `"off"` (current behavior), `"after"` (insert then speak text), and `"confirm"` (speak text, bind Space/Escape to cancel, auto-insert after configurable timeout). A shared `_clear_confirm_gestures` helper ensures gestures and timer are cleaned up exactly once regardless of which path resolves the confirm window.

**Tech Stack:** Python, NVDA add-on API (`ui`, `wx`, `globalPluginHandler`, `bindGesture`/`removeGestureBinding`), wxPython

---

## File Map

| File | Change |
|------|--------|
| `globalPlugins/groqVoiceDictation/config_manager.py` | Add `READBACK_MODES` constant; add `readbackMode` and `confirmTimeout` to `CONFSPEC` |
| `globalPlugins/groqVoiceDictation/settings_panel.py` | Add readback mode dropdown + confirm timeout spinner; wire `EVT_CHOICE` handler; update `onSave` |
| `globalPlugins/groqVoiceDictation/__init__.py` | Add instance attributes; add 4 new methods; modify `_process_recording`; modify `terminate` |

---

## Task 1: Add config keys and constants

**Files:**
- Modify: `globalPlugins/groqVoiceDictation/config_manager.py`

- [ ] **Step 1: Add `READBACK_MODES` constant after `FEEDBACK_MODES` (after line 28)**

  Open `config_manager.py`. After the `FEEDBACK_MODES` list (line 28), insert:

  ```python
  READBACK_MODES = [
      ("off", "Off"),
      ("after", "Read back after insertion"),
      ("confirm", "Read back and confirm before insertion"),
  ]
  ```

- [ ] **Step 2: Add two keys to `CONFSPEC` (after `"allowPasteFallback"` entry, line 41)**

  In the `CONFSPEC` dict, after the `"allowPasteFallback"` line, add:

  ```python
  "readbackMode": 'string(default="off")',
  "confirmTimeout": "integer(default=5,min=2,max=15)",
  ```

  The full `CONFSPEC` dict should now end with:

  ```python
  CONFSPEC = {
      "apiKey": 'string(default="")',
      "transcriptionModel": 'string(default="whisper-large-v3-turbo")',
      "cleanupMode": 'string(default="light")',
      "cleanupModel": 'string(default="llama-3.1-8b-instant")',
      "microphoneDevice": "integer(default=-1,min=-1,max=9999)",
      "silenceDetection": "boolean(default=true)",
      "silenceTimeout": "integer(default=2,min=1,max=15)",
      "feedbackMode": 'string(default="both")',
      "allowPasteFallback": "boolean(default=true)",
      "silenceThreshold": "integer(default=1500,min=100,max=32767)",
      "readbackMode": 'string(default="off")',
      "confirmTimeout": "integer(default=5,min=2,max=15)",
  }
  ```

- [ ] **Step 3: Commit**

  ```
  git add globalPlugins/groqVoiceDictation/config_manager.py
  git commit -m "feat: add readbackMode and confirmTimeout config keys"
  ```

---

## Task 2: Add settings UI controls

**Files:**
- Modify: `globalPlugins/groqVoiceDictation/settings_panel.py`

- [ ] **Step 1: Add readback mode dropdown and confirm timeout spinner in `makeSettings`**

  After the `self.allow_paste_fallback` block (after line 113) and before `self.silence_threshold`, insert:

  ```python
  self.readback_mode = sizer_helper.addLabeledControl(
      _("&Read-back mode:"),
      wx.Choice,
      choices=config_manager.label_list(config_manager.READBACK_MODES),
  )
  self.readback_mode.SetSelection(
      config_manager.index_for_value(config_manager.READBACK_MODES, conf["readbackMode"])
  )

  self.confirm_timeout = sizer_helper.addLabeledControl(
      _("Confirm &timeout (seconds):"),
      nvdaControls.SelectOnFocusSpinCtrl,
      value=str(conf["confirmTimeout"]),
      min=2,
      max=15,
  )
  self.confirm_timeout.Enable(conf["readbackMode"] == "confirm")
  ```

- [ ] **Step 2: Bind `EVT_CHOICE` for the readback mode dropdown**

  In `makeSettings`, after the existing `self.Bind` calls (after line 128), add:

  ```python
  self.Bind(wx.EVT_CHOICE, self.on_readback_mode_change, self.readback_mode)
  ```

- [ ] **Step 3: Add `on_readback_mode_change` handler method**

  After the `on_get_api_key` method (after line 224), add:

  ```python
  def on_readback_mode_change(self, _event) -> None:
      mode = config_manager.READBACK_MODES[self.readback_mode.GetSelection()][0]
      self.confirm_timeout.Enable(mode == "confirm")
  ```

- [ ] **Step 4: Add readback keys to `onSave`**

  In `onSave`, inside the `values` dict (after the `"silenceThreshold"` entry, line 238), add:

  ```python
  "readbackMode": config_manager.READBACK_MODES[self.readback_mode.GetSelection()][0],
  "confirmTimeout": self.confirm_timeout.GetValue(),
  ```

  The full `values` dict in `onSave` should now be:

  ```python
  values: dict[str, Any] = {
      "apiKey": self.api_key.GetValue().strip(),
      "transcriptionModel": config_manager.TRANSCRIPTION_MODELS[self.transcription_model.GetSelection()],
      "cleanupMode": config_manager.CLEANUP_MODES[self.cleanup_mode.GetSelection()][0],
      "cleanupModel": config_manager.CLEANUP_MODELS[self.cleanup_model.GetSelection()],
      "microphoneDevice": self._microphone_choices[self.microphone_device.GetSelection()][0],
      "silenceDetection": self.silence_detection.GetValue(),
      "silenceTimeout": self.silence_timeout.GetValue(),
      "feedbackMode": config_manager.FEEDBACK_MODES[self.feedback_mode.GetSelection()][0],
      "allowPasteFallback": self.allow_paste_fallback.GetValue(),
      "silenceThreshold": self.silence_threshold.GetValue(),
      "readbackMode": config_manager.READBACK_MODES[self.readback_mode.GetSelection()][0],
      "confirmTimeout": self.confirm_timeout.GetValue(),
  }
  ```

- [ ] **Step 5: Manual smoke test**

  Reload NVDA (or install the dev addon). Open NVDA Settings → Groq Voice Dictation.
  - Verify "Read-back mode" dropdown appears with three choices: Off, Read back after insertion, Read back and confirm before insertion.
  - Verify "Confirm timeout (seconds)" spinner is disabled when Off or After is selected.
  - Select "Read back and confirm before insertion" — spinner should enable.
  - Change back to Off — spinner should disable.
  - Save settings and reopen — selections should persist.

- [ ] **Step 6: Commit**

  ```
  git add globalPlugins/groqVoiceDictation/settings_panel.py
  git commit -m "feat: add readback mode and confirm timeout settings UI"
  ```

---

## Task 3: Add new methods to GlobalPlugin

**Files:**
- Modify: `globalPlugins/groqVoiceDictation/__init__.py`

- [ ] **Step 1: Add three new instance attributes in `__init__`**

  In `GlobalPlugin.__init__`, after `self._state_lock = threading.Lock()` (line 46), add:

  ```python
  self._pending_text: str | None = None
  self._confirm_timer: wx.CallLater | None = None
  self._confirm_gestures_bound: bool = False
  ```

- [ ] **Step 2: Add `_clear_confirm_gestures` method**

  After the `_notify` method (after line 170), add:

  ```python
  def _clear_confirm_gestures(self) -> None:
      if not self._confirm_gestures_bound:
          return
      self._confirm_gestures_bound = False
      self.removeGestureBinding("kb:space")
      self.removeGestureBinding("kb:escape")
      if self._confirm_timer is not None:
          self._confirm_timer.Stop()
          self._confirm_timer = None
  ```

- [ ] **Step 3: Add `_start_confirm_window` method**

  After `_clear_confirm_gestures`, add:

  ```python
  def _start_confirm_window(self, text: str) -> None:
      conf = config_manager.get()
      self._pending_text = text
      ui.message(text)
      self.bindGesture("kb:space", "cancelPendingDictation")
      self.bindGesture("kb:escape", "cancelPendingDictation")
      self._confirm_gestures_bound = True
      self._confirm_timer = wx.CallLater(
          conf["confirmTimeout"] * 1000, self._execute_pending_insert
      )
  ```

- [ ] **Step 4: Add `script_cancelPendingDictation` method**

  After `_start_confirm_window`, add:

  ```python
  def script_cancelPendingDictation(self, gesture) -> None:
      self._clear_confirm_gestures()
      self._pending_text = None
      with self._state_lock:
          self._processing = False
      self._notify(_("Dictation cancelled."))
  ```

  Note: no `@script` decorator — this method is bound dynamically via `bindGesture`, not statically.

- [ ] **Step 5: Add `_execute_pending_insert` method**

  After `script_cancelPendingDictation`, add:

  ```python
  def _execute_pending_insert(self) -> None:
      self._clear_confirm_gestures()
      text = self._pending_text
      self._pending_text = None
      conf = config_manager.get()
      inserted = self._text_inserter.insert(text, conf["allowPasteFallback"])
      with self._state_lock:
          self._processing = False
      if inserted:
          self._notify(_("Dictation inserted."), tone=980)
      else:
          self._notify(
              _("Could not insert the dictated text into the current control."),
              tone=220,
              is_error=True,
          )
  ```

- [ ] **Step 6: Commit**

  ```
  git add globalPlugins/groqVoiceDictation/__init__.py
  git commit -m "feat: add confirm window methods and _clear_confirm_gestures"
  ```

---

## Task 4: Wire readback into `_process_recording` and `terminate`

**Files:**
- Modify: `globalPlugins/groqVoiceDictation/__init__.py`

- [ ] **Step 1: Add `_confirm_pending` local variable at the top of `_process_recording`**

  In `_process_recording` (line 128), add `_confirm_pending = False` as the first line of the method body, before the `try:` block:

  ```python
  def _process_recording(self, wav_path: str) -> None:
      conf = config_manager.get()
      client = GroqClient(api_key=conf["apiKey"])
      _confirm_pending = False
      try:
  ```

- [ ] **Step 2: Replace the insertion block at the end of the `try` body**

  Replace lines 148–152 (the current insertion block):

  ```python
  inserted = self._text_inserter.insert(final_text, conf["allowPasteFallback"])
  if inserted:
      self._notify(_("Dictation inserted."), tone=980)
  else:
      self._notify(_("Could not insert the dictated text into the current control."), tone=220, is_error=True)
  ```

  With:

  ```python
  readback_mode = conf["readbackMode"]
  if readback_mode == "confirm":
      _confirm_pending = True
      wx.CallAfter(self._start_confirm_window, final_text)
  else:
      inserted = self._text_inserter.insert(final_text, conf["allowPasteFallback"])
      if inserted:
          self._notify(_("Dictation inserted."), tone=980)
          if readback_mode == "after":
              wx.CallAfter(ui.message, final_text)
      else:
          self._notify(
              _("Could not insert the dictated text into the current control."),
              tone=220,
              is_error=True,
          )
  ```

- [ ] **Step 3: Guard the `finally` block so it skips `_processing = False` in confirm mode**

  Replace lines 159–162 (the current `finally` block):

  ```python
  finally:
      AudioRecorder.delete_file(wav_path)
      with self._state_lock:
          self._processing = False
  ```

  With:

  ```python
  finally:
      AudioRecorder.delete_file(wav_path)
      if not _confirm_pending:
          with self._state_lock:
              self._processing = False
  ```

  The complete `_process_recording` method should now read:

  ```python
  def _process_recording(self, wav_path: str) -> None:
      conf = config_manager.get()
      client = GroqClient(api_key=conf["apiKey"])
      _confirm_pending = False
      try:
          self._notify(_("Transcribing."), tone=520)
          transcript = client.transcribe(wav_path, conf["transcriptionModel"])
          if not transcript.strip():
              self._notify(_("No speech was detected."), tone=260, is_error=True)
              return
          final_text = transcript
          if conf["cleanupMode"] != "raw":
              try:
                  final_text = client.cleanup(transcript, conf["cleanupMode"], conf["cleanupModel"])
              except GroqClientError as exc:
                  log.error("Groq cleanup failed: %s (%s)", exc.message, exc.category)
                  self._notify(_("Cleanup failed. Inserting the raw transcript."), tone=420, is_error=True)
                  final_text = transcript
          if not final_text.strip():
              self._notify(_("The cleanup step returned empty text."), tone=260, is_error=True)
              return
          readback_mode = conf["readbackMode"]
          if readback_mode == "confirm":
              _confirm_pending = True
              wx.CallAfter(self._start_confirm_window, final_text)
          else:
              inserted = self._text_inserter.insert(final_text, conf["allowPasteFallback"])
              if inserted:
                  self._notify(_("Dictation inserted."), tone=980)
                  if readback_mode == "after":
                      wx.CallAfter(ui.message, final_text)
              else:
                  self._notify(
                      _("Could not insert the dictated text into the current control."),
                      tone=220,
                      is_error=True,
                  )
      except GroqClientError as exc:
          log.error("Groq dictation failed: %s (%s)", exc.message, exc.category)
          self._notify(exc.message, tone=220, is_error=True)
      except Exception:
          log.exception("Unexpected Groq Voice Dictation failure")
          self._notify(_("Unexpected dictation error. Check the NVDA log for details."), tone=220, is_error=True)
      finally:
          AudioRecorder.delete_file(wav_path)
          if not _confirm_pending:
              with self._state_lock:
                  self._processing = False
  ```

- [ ] **Step 4: Add cleanup to `terminate`**

  In `terminate` (line 48), add two lines at the very start of the method body, before `with self._state_lock:`:

  ```python
  def terminate(self):
      self._clear_confirm_gestures()
      self._pending_text = None
      with self._state_lock:
          recorder = self._recorder
          self._recorder = None
      ...
  ```

- [ ] **Step 5: Manual test — "after" mode**

  - In NVDA Settings, set Read-back mode to "Read back after insertion".
  - Focus a text field. Press NVDA+Shift+V, dictate a sentence, stop.
  - NVDA should say "Dictation inserted." and then speak the inserted text.
  - Confirm the text appears in the field.

- [ ] **Step 6: Manual test — "confirm" mode, let it auto-insert**

  - Set Read-back mode to "Read back and confirm before insertion", timeout 5 seconds.
  - Focus a text field. Dictate a sentence.
  - NVDA should read the transcribed text.
  - Do not press any key. After 5 seconds the text should be inserted and NVDA says "Dictation inserted."
  - Confirm the text appears in the field.

- [ ] **Step 7: Manual test — "confirm" mode, cancel with Escape**

  - Dictate a sentence with confirm mode active.
  - While NVDA is reading the text back, press Escape.
  - NVDA should say "Dictation cancelled." and nothing should be inserted.
  - Confirm the text field is unchanged.

- [ ] **Step 8: Manual test — "confirm" mode, cancel with Space**

  - Repeat Step 7 using Space instead of Escape.
  - Same expected result: "Dictation cancelled.", field unchanged.

- [ ] **Step 9: Manual test — new dictation blocked during confirm window**

  - Start a dictation with confirm mode. While the confirm window is open, press NVDA+Shift+V.
  - NVDA should say "Still processing the previous dictation." — no new recording starts.

- [ ] **Step 10: Manual test — "off" mode unchanged**

  - Set Read-back mode to Off.
  - Dictate normally. NVDA should say "Dictation inserted." with no extra speech. Text appears in field.

- [ ] **Step 11: Commit**

  ```
  git add globalPlugins/groqVoiceDictation/__init__.py
  git commit -m "feat: wire readback modes into _process_recording and terminate"
  ```

---

## Task 5: Rebuild addon

**Files:**
- Modify: `build_addon.ps1` (version bump if releasing)
- Output: `dist/groqVoiceDictation-0.2.0.nvda-addon`

- [ ] **Step 1: Run the build script**

  ```powershell
  .\build_addon.ps1
  ```

  Expected output:
  ```
  Built C:\Users\willb\programs\nvda voice input\dist\groqVoiceDictation-0.2.0.nvda-addon
  ```

- [ ] **Step 2: Install in NVDA for final integration test**

  In NVDA: Tools → Manage Add-ons → Install → select `dist/groqVoiceDictation-0.2.0.nvda-addon`. Restart NVDA when prompted.

- [ ] **Step 3: Full end-to-end test after install**

  Repeat the manual tests from Task 4, Steps 5–10, with the installed addon (not scratchpad) to confirm the packaged build works correctly.

- [ ] **Step 4: Commit built artifact**

  ```
  git add dist/groqVoiceDictation-0.2.0.nvda-addon
  git commit -m "build: rebuild addon with readback feature"
  ```
