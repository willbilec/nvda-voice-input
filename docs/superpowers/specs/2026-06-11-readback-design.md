# Read-Back Feature Design

**Date:** 2026-06-11
**Status:** Approved

## Overview

Add an option for NVDA to read back the final transcribed (and optionally cleaned-up) text after dictation. Two modes are supported: read-after-insert (informational feedback) and read-then-confirm (intercepts insertion pending user cancellation or timeout).

## Configuration

Two new keys in `CONFSPEC`:

- `readbackMode`: `string(default="off")` — one of `"off"`, `"after"`, `"confirm"`
- `confirmTimeout`: `integer(default=5, min=2, max=15)` — seconds before auto-insert in confirm mode

New constant in `config_manager.py`:

```python
READBACK_MODES = [
    ("off",     "Off"),
    ("after",   "Read back after insertion"),
    ("confirm", "Read back and confirm before insertion"),
]
```

## Settings Panel

- Add **Read-back mode** dropdown (`wx.Choice`) populated from `READBACK_MODES`.
- Add **Confirm timeout (seconds)** spinner (`nvdaControls.SelectOnFocusSpinCtrl`, range 2–15), enabled only when `confirm` is selected. Bind `wx.EVT_CHOICE` on the dropdown to toggle the spinner's enabled state.

## Plugin Logic (`__init__.py`)

### New instance attributes

```python
self._pending_text: str | None = None
self._confirm_timer: wx.CallLater | None = None
self._confirm_gestures_bound: bool = False
```

### `"after"` mode

In `_process_recording`, after successful insertion, add:

```python
wx.CallAfter(ui.message, final_text)
```

This runs after the existing "Dictation inserted." notify.

### `"confirm"` mode

In `_process_recording`, instead of calling `_text_inserter.insert`, call:

```python
wx.CallAfter(self._start_confirm_window, final_text)
```

**`_start_confirm_window(self, text: str)`** (main thread):
1. Store `text` in `self._pending_text`
2. Speak text via `ui.message(text)`
3. Bind `kb:space` and `kb:escape` to `"cancelPendingDictation"`
4. Set `self._confirm_gestures_bound = True`
5. Start `self._confirm_timer = wx.CallLater(conf["confirmTimeout"] * 1000, self._execute_pending_insert)`

**`script_cancelPendingDictation(self, gesture)`** (Space or Escape pressed — no `@script` decorator; gestures are bound dynamically via `bindGesture`, not statically):
1. `self._clear_confirm_gestures()`
2. `self._pending_text = None`
3. `with self._state_lock: self._processing = False`
4. `self._notify(_("Dictation cancelled."))`

**`_execute_pending_insert(self)`** (timer fires, main thread):
1. `self._clear_confirm_gestures()`
2. Retrieve and clear `_pending_text`
3. Read `conf["allowPasteFallback"]`
4. Call `self._text_inserter.insert(text, allow_paste_fallback)`
5. `with self._state_lock: self._processing = False`
6. Notify "Dictation inserted." on success or "Could not insert..." on failure

**`_clear_confirm_gestures(self)`** (shared cleanup, main thread):

```python
def _clear_confirm_gestures(self):
    if not self._confirm_gestures_bound:
        return
    self._confirm_gestures_bound = False  # set first
    self.removeGestureBinding("kb:space")
    self.removeGestureBinding("kb:escape")
    if self._confirm_timer is not None:
        self._confirm_timer.Stop()
        self._confirm_timer = None
```

Both `script_cancelPendingDictation` and `_execute_pending_insert` call this first. Because both run on the wx main thread, the flag is an effective single-entry guard with no threading race.

### `terminate()` cleanup

Before calling `super().terminate()`, add:

```python
self._clear_confirm_gestures()
self._pending_text = None
```

## Edge Cases

- **New dictation during confirm window:** `_processing` remains `True`, so the existing "Still processing" guard fires — no special handling needed.
- **Insertion fails in confirm mode:** notify "Could not insert the dictated text into the current control." and set `_processing = False`, matching existing behavior.
- **Plugin shutdown with open confirm window:** `terminate()` cleanup discards pending text without inserting.

## Files Changed

| File | Change |
|------|--------|
| `config_manager.py` | Add `READBACK_MODES`, `readbackMode`, `confirmTimeout` to `CONFSPEC` |
| `settings_panel.py` | Add readback mode dropdown + confirm timeout spinner |
| `__init__.py` | Add confirm window logic, cancel script, after-insert readback, terminate cleanup |
