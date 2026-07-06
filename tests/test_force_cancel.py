"""Tests for the double-press escape hatch and force-cancel flow.

These tests exercise the GlobalPlugin's double-press detection, the
force-cancel method, and the cancel-token machinery that lets an
in-flight processing worker bail out before inserting text.

NVDA modules (addonHandler, globalPluginHandler, gui, wx, etc.) are
stubbed out in setUpModule so the plugin module can be imported in
isolation. The GlobalPlugin instance is created via ``__new__`` to
bypass ``__init__`` (which touches NVDA state); the tests then
populate the attributes the new code paths read.
"""

import importlib
import pathlib
import sys
import threading
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "globalPlugins" / "groqVoiceDictation"
PACKAGE_PARENT = MODULE_DIR.parent
PLUGIN_PKG = "globalPlugins.groqVoiceDictation"


def _install_nvda_stubs() -> None:
	"""Install minimal NVDA-module stubs into sys.modules so the plugin
	package can be imported without NVDA being installed.
	"""
	if "addonHandler" not in sys.modules:
		_stub_addon = types.ModuleType("addonHandler")
		_stub_addon.AddonError = type("AddonError", (Exception,), {})

		def _stub_init_translation():
			# NVDA's real initTranslation installs the gettext _ function
			# into builtins so module-level code can call _("string").
			# In tests we don't need translation, so just identity-wrap.
			import builtins
			builtins.__dict__.setdefault("_", lambda s: s)

		_stub_addon.initTranslation = _stub_init_translation
		_stub_addon.getCodeAddon = lambda: types.SimpleNamespace(
			manifest={"summary": "Groq Voice Dictation"}
		)
		sys.modules["addonHandler"] = _stub_addon

	if "globalPluginHandler" not in sys.modules:
		_stub_gph = types.ModuleType("globalPluginHandler")
		_stub_gph.GlobalPlugin = type("GlobalPlugin", (), {"__init__": lambda self: None})
		sys.modules["globalPluginHandler"] = _stub_gph

	if "gui" not in sys.modules:
		_stub_gui = types.ModuleType("gui")
		_stub_gui.guiHelper = types.SimpleNamespace()
		_stub_gui.nvdaControls = types.SimpleNamespace()
		_stub_settings = types.ModuleType("gui.settingsDialogs")
		_stub_settings.NVDASettingsDialog = types.SimpleNamespace(categoryClasses=[])
		_stub_settings.SettingsPanel = type("SettingsPanel", (), {})
		_stub_gui.settingsDialogs = _stub_settings
		sys.modules["gui"] = _stub_gui
		sys.modules["gui.settingsDialogs"] = _stub_settings
		sys.modules["gui.guiHelper"] = _stub_gui.guiHelper
		sys.modules["gui.nvdaControls"] = _stub_gui.nvdaControls

	if "tones" not in sys.modules:
		_stub_tones = types.ModuleType("tones")
		_stub_tones.beep = lambda *args, **kwargs: None
		sys.modules["tones"] = _stub_tones

	if "ui" not in sys.modules:
		_stub_ui = types.ModuleType("ui")
		_stub_ui.message = lambda *args, **kwargs: None
		sys.modules["ui"] = _stub_ui

	if "wx" not in sys.modules:
		_stub_wx = types.ModuleType("wx")
		_stub_wx.CallLater = type(
			"CallLater",
			(),
			{
				"__init__": lambda self, *args, **kwargs: None,
				"Stop": lambda self: None,
			},
		)
		_stub_wx.CallAfter = lambda fn, *args, **kwargs: fn(*args)
		sys.modules["wx"] = _stub_wx

	if "logHandler" not in sys.modules:
		_stub_log = types.ModuleType("logHandler")
		_stub_log.log = types.SimpleNamespace(
			info=lambda *a, **k: None,
			warning=lambda *a, **k: None,
			error=lambda *a, **k: None,
			exception=lambda *a, **k: None,
			debug=lambda *a, **k: None,
		)
		sys.modules["logHandler"] = _stub_log
	else:
		# Make sure subsequent test runs in the same process see the
		# full surface (an earlier runner may have installed a partial
		# stub). Set defaults without clobbering an existing attribute.
		_existing = sys.modules["logHandler"].log
		for _name in ("info", "warning", "error", "exception", "debug"):
			if not hasattr(_existing, _name):
				setattr(_existing, _name, lambda *a, **k: None)

	if "scriptHandler" not in sys.modules:
		_stub_script = types.ModuleType("scriptHandler")
		_stub_script.script = lambda *dargs, **dkwargs: (lambda fn: fn)
		sys.modules["scriptHandler"] = _stub_script

	if "config" not in sys.modules:
		# config_manager.get() does `config.conf[SECTION]`, so conf has to
		# be subscriptable. A real dict covers that; we also need .spec and
		# .profiles for other code paths.
		_conf_store: dict = {}
		_stub_conf = {
			"spec": {},
			"profiles": [{}],
		}
		# __getitem__/__setitem__ on dict already work for normal keys,
		# but the plugin uses config.conf[SECTION] = value too, and the
		# default dict will KeyError on missing keys. Wrap with a defaultdict
		# to be safe.
		from collections import defaultdict
		_conf_inner: dict = defaultdict(dict)
		_conf_inner.update(_stub_conf)
		_conf_inner.setdefault("profiles", [{}])
		sys.modules["config"] = types.SimpleNamespace(
			conf=_conf_inner,
			AggregatedSection=dict,
		)

	if "api" not in sys.modules:
		sys.modules["api"] = types.ModuleType("api")
	if "core" not in sys.modules:
		sys.modules["core"] = types.ModuleType("core")
	if "watchdog" not in sys.modules:
		sys.modules["watchdog"] = types.ModuleType("watchdog")
	if "keyboardHandler" not in sys.modules:
		_stub_kbh = types.ModuleType("keyboardHandler")
		_stub_kbh.KeyboardInputGesture = type("KeyboardInputGesture", (), {})
		sys.modules["keyboardHandler"] = _stub_kbh

	# Stub settings_panel wholesale — it pulls in wx.Dialog subclasses and
	# dialog-builder code that is irrelevant to these tests. The plugin
	# only references GroqVoiceDictationSettingsPanel at module import,
	# so a SimpleNamespace with that one attribute is enough.
	if "globalPlugins.groqVoiceDictation.settings_panel" not in sys.modules:
		_stub_sp = types.ModuleType("globalPlugins.groqVoiceDictation.settings_panel")
		_stub_sp.GroqVoiceDictationSettingsPanel = type(
			"GroqVoiceDictationSettingsPanel", (), {}
		)
		sys.modules["globalPlugins.groqVoiceDictation.settings_panel"] = _stub_sp

	# audio_recorder imports pyaudio; text_inserter imports a long list of
	# NVDA modules — both have to be stubbed.
	if "pyaudio" not in sys.modules:
		sys.modules["pyaudio"] = types.SimpleNamespace(
			PyAudio=type("PyAudio", (), {}),
			paInt16=8,
			paContinue=0,
		)

	# Ensure the package init path is set up.
	if str(MODULE_DIR) not in sys.path:
		sys.path.insert(0, str(MODULE_DIR))
	if str(MODULE_DIR / "lib") not in sys.path:
		sys.path.insert(0, str(MODULE_DIR / "lib"))


def _load_plugin_module():
	"""Import (or reload) the GlobalPlugin module with NVDA stubs in place.

	The `globalPlugins` directory has no ``__init__.py`` in this repo, so
	Python treats it as a namespace package. We make the parent of
	`globalPlugins` importable and let normal package import do the rest.
	"""
	_install_nvda_stubs()
	parent = str(PACKAGE_PARENT.parent)
	if parent not in sys.path:
		sys.path.insert(0, parent)
	# If a previous test run already registered these, drop them so the
	# relative imports inside the plugin re-resolve against fresh stubs.
	for mod_name in (PLUGIN_PKG, PACKAGE_PARENT.name):
		sys.modules.pop(mod_name, None)
	# Re-install the settings_panel stub in case a previous run overwrote it.
	_stub_sp = types.ModuleType("globalPlugins.groqVoiceDictation.settings_panel")
	_stub_sp.GroqVoiceDictationSettingsPanel = type(
		"GroqVoiceDictationSettingsPanel", (), {}
	)
	sys.modules["globalPlugins.groqVoiceDictation.settings_panel"] = _stub_sp
	import globalPlugins  # noqa: F401  -- namespace package; registers itself
	import globalPlugins.groqVoiceDictation  # noqa: F401
	return importlib.reload(sys.modules[PLUGIN_PKG])


_install_nvda_stubs()
importlib.util  # noqa: E402  -- referenced above
_PLUGIN_MODULE = _load_plugin_module()


def _make_plugin():
	"""Build a GlobalPlugin instance without invoking its real __init__.

	The real __init__ touches NVDA state (NVDASettingsDialog,
	ensure_config_spec, etc.); the tests don't need any of that.
	"""
	instance = object.__new__(_PLUGIN_MODULE.GlobalPlugin)
	instance._recorder = None
	instance._processing = False
	instance._text_inserter = mock.MagicMock()
	instance._state_lock = threading.Lock()
	instance._pending_text = None
	instance._confirm_timer = None
	instance._confirm_gestures_bound = False
	instance._preflight_timer = None
	instance._last_toggle_time = 0.0
	instance._double_press_window_ms = 500
	instance._cancel_token = 0
	instance._notify = mock.MagicMock()
	# The real base class provides these; we instantiate via __new__ so
	# the plugin instance doesn't get them automatically. Tests that need
	# to assert on gesture-binding behavior install mocks here.
	instance.removeGestureBinding = mock.MagicMock()
	instance.bindGesture = mock.MagicMock()
	return instance


class DoublePressDetectionTests(unittest.TestCase):
	"""Verify the toggle script's double-press detection logic."""

	def setUp(self) -> None:
		self.plugin = _make_plugin()

	def test_first_press_is_normal_toggle(self):
		"""A single press with no prior press history should do the
		normal toggle (start recording when idle), not a force-cancel.
		"""
		with mock.patch.object(self.plugin, "_start_recording") as start, \
				mock.patch.object(self.plugin, "_stop_and_process") as stop, \
				mock.patch.object(self.plugin, "_force_cancel") as force:
			self.plugin.script_toggleVoiceDictation(None)
		start.assert_called_once()
		stop.assert_not_called()
		force.assert_not_called()
		self.assertGreater(self.plugin._last_toggle_time, 0.0)

	def test_two_quick_presses_trigger_force_cancel(self):
		"""Two presses within the double-press window should fire
		_force_cancel, not the normal toggle for the second press.
		"""
		with mock.patch.object(self.plugin, "_start_recording") as start, \
				mock.patch.object(self.plugin, "_stop_and_process") as stop, \
				mock.patch.object(self.plugin, "_force_cancel") as force:
			self.plugin.script_toggleVoiceDictation(None)  # 1st press
			self.plugin.script_toggleVoiceDictation(None)  # 2nd press, quick
		start.assert_called_once()
		stop.assert_not_called()
		force.assert_called_once()

	def test_second_press_after_window_is_normal_toggle(self):
		"""Two presses separated by more than the window should both
		be normal toggles, not a force-cancel.
		"""
		# Pretend the first press was a long time ago.
		self.plugin._last_toggle_time = 0.0
		# _start_recording is the method the toggle calls when idle. Use
		# side_effect to actually populate _recorder so the second press
		# sees the recording state and routes to _stop_and_process.
		def fake_start():
			rec = mock.MagicMock()
			rec.is_recording = True
			self.plugin._recorder = rec
		with mock.patch("time.monotonic", side_effect=[100.0, 100.0 + 1.0]), \
				mock.patch.object(self.plugin, "_start_recording", side_effect=fake_start) as start, \
				mock.patch.object(self.plugin, "_stop_and_process") as stop, \
				mock.patch.object(self.plugin, "_force_cancel") as force:
			self.plugin.script_toggleVoiceDictation(None)
			# 1 second > 500 ms window
			self.plugin.script_toggleVoiceDictation(None)
		# Both presses were normal toggles: first starts recording,
		# second stops (because _recorder is not None per the branch above).
		start.assert_called_once()
		stop.assert_called_once()
		force.assert_not_called()

	def test_double_press_resets_timer_for_subsequent_press(self):
		"""After a force-cancel, the next single press should be a
		normal toggle, not another force-cancel.
		"""
		with mock.patch.object(self.plugin, "_start_recording") as start, \
				mock.patch.object(self.plugin, "_stop_and_process") as stop, \
				mock.patch.object(self.plugin, "_force_cancel") as force:
			self.plugin.script_toggleVoiceDictation(None)  # normal: start
			self.plugin.script_toggleVoiceDictation(None)  # double-press: cancel
			self.plugin.script_toggleVoiceDictation(None)  # normal again: start
		self.assertEqual(start.call_count, 2)
		self.assertEqual(force.call_count, 1)
		stop.assert_not_called()

	def test_processing_state_blocks_single_press(self):
		"""A single press while processing should not start a new
		recording or fire a force-cancel; the user must double-press
		to escape.
		"""
		self.plugin._processing = True
		with mock.patch.object(self.plugin, "_start_recording") as start, \
				mock.patch.object(self.plugin, "_stop_and_process") as stop, \
				mock.patch.object(self.plugin, "_force_cancel") as force:
			self.plugin.script_toggleVoiceDictation(None)
		start.assert_not_called()
		stop.assert_not_called()
		force.assert_not_called()
		self.plugin._notify.assert_called()

	def test_double_press_escapes_processing_state(self):
		"""The whole point of the feature: two quick presses should
		force-cancel out of a processing state.
		"""
		self.plugin._processing = True
		with mock.patch.object(self.plugin, "_start_recording") as start, \
				mock.patch.object(self.plugin, "_stop_and_process") as stop, \
				mock.patch.object(self.plugin, "_force_cancel") as force:
			self.plugin.script_toggleVoiceDictation(None)  # 1st: blocked
			self.plugin.script_toggleVoiceDictation(None)  # 2nd: force-cancel
		start.assert_not_called()
		stop.assert_not_called()
		force.assert_called_once()

	def test_second_press_during_recording_triggers_force_cancel(self):
		"""Two quick presses should abort a recording-in-progress
		instead of stopping and processing it.
		"""
		recorder = mock.MagicMock()
		recorder.is_recording = True
		self.plugin._recorder = recorder
		with mock.patch.object(self.plugin, "_stop_and_process") as stop, \
				mock.patch.object(self.plugin, "_force_cancel") as force:
			self.plugin.script_toggleVoiceDictation(None)  # 1st: would stop+process
			self.plugin.script_toggleVoiceDictation(None)  # 2nd: force-cancel
		# First press: not yet processing, recorder is recording,
		# so it called _stop_and_process (normal toggle).
		stop.assert_called_once()
		force.assert_called_once()


class ForceCancelTests(unittest.TestCase):
	"""Verify _force_cancel tears everything down correctly."""

	def setUp(self) -> None:
		self.plugin = _make_plugin()

	def test_force_cancel_bumps_token(self):
		original = self.plugin._cancel_token
		self.plugin._force_cancel()
		self.assertEqual(self.plugin._cancel_token, original + 1)

	def test_force_cancel_clears_processing_flag(self):
		self.plugin._processing = True
		self.plugin._force_cancel()
		self.assertFalse(self.plugin._processing)

	def test_force_cancel_clears_pending_text(self):
		self.plugin._pending_text = "should be discarded"
		self.plugin._force_cancel()
		self.assertIsNone(self.plugin._pending_text)

	def test_force_cancel_stops_recorder_and_deletes_wav(self):
		recorder = mock.MagicMock()
		recorder.is_recording = True
		recorder.stop.return_value = "/tmp/recorded.wav"
		self.plugin._recorder = recorder
		with mock.patch.object(_PLUGIN_MODULE, "AudioRecorder") as ar:
			ar.delete_file = mock.MagicMock()
			self.plugin._force_cancel()
		recorder.stop.assert_called_once()
		ar.delete_file.assert_called_once_with("/tmp/recorded.wav")
		self.assertIsNone(self.plugin._recorder)

	def test_force_cancel_handles_recorder_stop_exception(self):
		recorder = mock.MagicMock()
		recorder.is_recording = True
		recorder.stop.side_effect = Exception("boom")
		self.plugin._recorder = recorder
		# Should not raise.
		self.plugin._force_cancel()
		self.assertIsNone(self.plugin._recorder)

	def test_force_cancel_cancels_preflight(self):
		timer = mock.MagicMock()
		self.plugin._preflight_timer = timer
		self.plugin._force_cancel()
		timer.Stop.assert_called_once()
		self.assertIsNone(self.plugin._preflight_timer)

	def test_force_cancel_clears_confirm_gestures(self):
		self.plugin._confirm_gestures_bound = True
		self.plugin._confirm_timer = mock.MagicMock()
		with mock.patch.object(self.plugin, "removeGestureBinding") as remove:
			self.plugin._force_cancel()
		remove.assert_any_call("kb:space")
		remove.assert_any_call("kb:escape")
		self.assertFalse(self.plugin._confirm_gestures_bound)

	def test_force_cancel_safe_when_idle(self):
		# No recorder, no confirm, no preflight — should be a no-op
		# for those subsystems, but still bump the token and notify.
		self.plugin._force_cancel()
		self.assertEqual(self.plugin._cancel_token, 1)
		self.assertFalse(self.plugin._processing)
		self.plugin._notify.assert_called_once()


class CancelTokenTests(unittest.TestCase):
	"""Verify _is_cancelled and the worker bail-out logic."""

	def setUp(self) -> None:
		self.plugin = _make_plugin()

	def test_is_cancelled_false_for_current_token(self):
		token = self.plugin._cancel_token
		self.assertFalse(self.plugin._is_cancelled(token))

	def test_is_cancelled_true_after_force_cancel(self):
		token = self.plugin._cancel_token
		self.plugin._force_cancel()
		self.assertTrue(self.plugin._is_cancelled(token))

	def test_is_cancelled_false_for_new_token_after_force_cancel(self):
		"""A new worker starting after a force-cancel should see its
		own token as 'current' — that's the whole reason we use a
		counter instead of a boolean.
		"""
		self.plugin._force_cancel()
		new_token = self.plugin._cancel_token
		self.assertFalse(self.plugin._is_cancelled(new_token))

	def test_worker_bails_out_when_cancelled(self):
		"""If the token is bumped between worker start and the first
		check, the worker should return immediately without calling
		the transcribe API.
		"""
		# Simulate the worker capturing the token, then a force-cancel
		# arriving before the first check.
		captured = self.plugin._cancel_token
		self.plugin._force_cancel()
		# The worker checks the flag at the top of its try block.
		self.assertTrue(self.plugin._is_cancelled(captured))
		# And the finally clause must NOT clobber a fresher worker's
		# processing flag (a brand new dictation may have started).
		# Simulate that by leaving _processing set by something else
		# and verify we don't reset it.
		# (We can't actually test the finally without reimplementing
		# the whole method; this is mostly a smoke test of the API.)

	def test_new_worker_after_cancel_is_not_cancelled(self):
		"""The classic regression: a still-running old worker must
		see 'cancelled' but a brand-new worker must not.
		"""
		old_token = self.plugin._cancel_token
		self.plugin._force_cancel()
		# Old worker still in flight:
		self.assertTrue(self.plugin._is_cancelled(old_token))
		# User starts a new dictation; the new worker captures the
		# current token (which force-cancel just bumped):
		new_token = self.plugin._cancel_token
		self.assertFalse(self.plugin._is_cancelled(new_token))


class ExecutePendingInsertTests(unittest.TestCase):
	"""Verify the confirm-window path handles a force-cancel race."""

	def setUp(self) -> None:
		self.plugin = _make_plugin()

	def test_execute_pending_insert_with_none_text_does_not_crash(self):
		"""A force-cancel can clear _pending_text between the timer
		firing and the callback running. The insert method must not
		be called with None.
		"""
		self.plugin._pending_text = None
		# Pre-populate the inserter so we can assert it was NOT called.
		inserter = mock.MagicMock()
		inserter.insert.return_value = True
		self.plugin._text_inserter = inserter
		self.plugin._execute_pending_insert()
		inserter.insert.assert_not_called()
		# And _processing should be reset so the next toggle works.
		self.assertFalse(self.plugin._processing)

	def test_execute_pending_insert_inserts_text_normally(self):
		self.plugin._pending_text = "hello world"
		inserter = mock.MagicMock()
		inserter.insert.return_value = True
		self.plugin._text_inserter = inserter
		with mock.patch.object(_PLUGIN_MODULE.config_manager, "get") as get_conf:
			get_conf.return_value = {
				"allowPasteFallback": True,
				"feedbackMode": "tones",
			}
			self.plugin._execute_pending_insert()
		inserter.insert.assert_called_once_with("hello world", mock.ANY)
		self.assertFalse(self.plugin._processing)
		self.plugin._notify.assert_called()


if __name__ == "__main__":
	unittest.main()
