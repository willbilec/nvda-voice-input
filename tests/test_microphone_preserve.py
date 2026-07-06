"""Tests for the microphone-preservation fix in the settings panel.

The bug: ``list_input_devices`` filters out some host APIs (MME,
"Microsoft Sound Mapper", generic "Input"). When a user picked one of
those filtered devices, opening the microphone settings dialog would
silently show "System default" as selected (because
``_index_for_device`` returns index 0 for missing entries). Clicking
OK would then overwrite the user's saved device index with -1.

The fix: ``ensure_device_in_choices`` injects the user's saved device
as an explicit choice list entry so the dialog can show it as the
active selection. These tests pin the contract.
"""

import importlib
import pathlib
import sys
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "globalPlugins" / "groqVoiceDictation"
PACKAGE_PARENT = MODULE_DIR.parent
PLUGIN_PKG = "globalPlugins.groqVoiceDictation"


def _install_nvda_stubs() -> None:
	if "addonHandler" not in sys.modules:
		_stub = types.ModuleType("addonHandler")
		_stub.AddonError = type("AddonError", (Exception,), {})

		def _stub_init_translation():
			import builtins
			builtins.__dict__.setdefault("_", lambda s: s)
		_stub.initTranslation = _stub_init_translation
		_stub.getCodeAddon = lambda: types.SimpleNamespace(
			manifest={"summary": "Groq Voice Dictation"}
		)
		sys.modules["addonHandler"] = _stub
	if "globalPluginHandler" not in sys.modules:
		_stub = types.ModuleType("globalPluginHandler")
		_stub.GlobalPlugin = type("GlobalPlugin", (), {"__init__": lambda self: None})
		sys.modules["globalPluginHandler"] = _stub
	if "gui" not in sys.modules:
		_gui = types.ModuleType("gui")
		_gui.guiHelper = types.SimpleNamespace()
		_gui.nvdaControls = types.SimpleNamespace()
		_settings = types.ModuleType("gui.settingsDialogs")
		_settings.NVDASettingsDialog = types.SimpleNamespace(categoryClasses=[])
		_settings.SettingsPanel = type("SettingsPanel", (), {})
		_gui.settingsDialogs = _settings
		sys.modules["gui"] = _gui
		sys.modules["gui.settingsDialogs"] = _settings
		sys.modules["gui.guiHelper"] = _gui.guiHelper
		sys.modules["gui.nvdaControls"] = _gui.nvdaControls
	if "tones" not in sys.modules:
		sys.modules["tones"] = types.ModuleType("tones")
		sys.modules["tones"].beep = lambda *a, **k: None
	if "ui" not in sys.modules:
		sys.modules["ui"] = types.ModuleType("ui")
		sys.modules["ui"].message = lambda *a, **k: None
	# wx — the settings panel needs a long list of widget types. Always
	# re-install our full stub in case a previous test file put a
	# smaller one in sys.modules.
	_wx = types.ModuleType("wx")
	# Generic base so any "wx.X" reference resolves to an empty class.
	_WX_BASE = type("_WX_BASE", (), {})
	_wx_attrs = [
		"Window", "Dialog", "Panel", "Choice", "CheckBox", "Button",
		"StaticText", "BoxSizer", "TextCtrl", "MessageDialog", "MessageBox",
		"Size", "Sizer", "Font", "Colour",
	]
	for name in _wx_attrs:
		setattr(_wx, name, _WX_BASE)
	# Constants / sentinels used in the settings panel
	_wx.ID_OK = 1
	_wx.ID_YES = 2
	_wx.OK = 0x4000
	_wx.LEFT = 0
	_wx.EXPAND = 0
	_wx.ALIGN_CENTER_HORIZONTAL = 0
	_wx.TE_MULTILINE = 0
	_wx.EVT_BUTTON = "EVT_BUTTON"
	_wx.EVT_CHOICE = "EVT_CHOICE"
	# Used by the plugin itself
	_wx.CallLater = type(
		"CallLater", (),
		{"__init__": lambda self, *a, **k: None, "Stop": lambda self: None},
	)
	_wx.CallAfter = lambda fn, *a, **k: fn(*a)
	sys.modules["wx"] = _wx
	if "logHandler" not in sys.modules:
		_stub = types.ModuleType("logHandler")
		_stub.log = types.SimpleNamespace(
			info=lambda *a, **k: None,
			warning=lambda *a, **k: None,
			error=lambda *a, **k: None,
			exception=lambda *a, **k: None,
			debug=lambda *a, **k: None,
		)
		sys.modules["logHandler"] = _stub
	if "scriptHandler" not in sys.modules:
		sys.modules["scriptHandler"] = types.ModuleType("scriptHandler")
		sys.modules["scriptHandler"].script = lambda *d, **dk: (lambda fn: fn)
	if "config" not in sys.modules:
		from collections import defaultdict
		_conf = defaultdict(dict)
		_conf["spec"] = {}
		_conf["profiles"] = [{}]
		sys.modules["config"] = types.SimpleNamespace(
			conf=_conf, AggregatedSection=dict
		)
	if "api" not in sys.modules:
		sys.modules["api"] = types.ModuleType("api")
	if "core" not in sys.modules:
		sys.modules["core"] = types.ModuleType("core")
	if "watchdog" not in sys.modules:
		sys.modules["watchdog"] = types.ModuleType("watchdog")
	if "keyboardHandler" not in sys.modules:
		_stub = types.ModuleType("keyboardHandler")
		_stub.KeyboardInputGesture = type("KeyboardInputGesture", (), {})
		sys.modules["keyboardHandler"] = _stub
	if "pyaudio" not in sys.modules:
		sys.modules["pyaudio"] = types.SimpleNamespace(
			PyAudio=type("PyAudio", (), {}), paInt16=8, paContinue=0
		)

	# Do NOT stub settings_panel — we need the real module so that
	# `ensure_device_in_choices` is importable. The wx widget classes it
	# references (wx.Dialog, wx.Choice, etc.) are stubbed above as empty
	# type objects, which is enough to let the class definitions load.


def _load_plugin():
	_install_nvda_stubs()
	parent = str(PACKAGE_PARENT.parent)
	if parent not in sys.path:
		sys.path.insert(0, parent)
	# Drop any previous imports so relative imports re-resolve against
	# the fresh NVDA stubs and the real settings_panel.
	for mod_name in (PLUGIN_PKG, PACKAGE_PARENT.name):
		sys.modules.pop(mod_name, None)
	# settings_panel may have been stubbed by a previous test file; drop it
	# too so the plugin's `from .settings_panel import ...` resolves to
	# the real module.
	sys.modules.pop("globalPlugins.groqVoiceDictation.settings_panel", None)
	import globalPlugins  # noqa: F401
	import globalPlugins.groqVoiceDictation  # noqa: F401
	return importlib.reload(sys.modules[PLUGIN_PKG])


_plugin = _load_plugin()
ensure_device_in_choices = _plugin.settings_panel.ensure_device_in_choices


class EnsureDeviceInChoicesTests(unittest.TestCase):
	"""Pin the contract of the bug fix."""

	def test_missing_device_is_appended(self):
		choices = [(-1, "System default"), (3, "USB Mic")]
		added = ensure_device_in_choices(choices, 25, label="Custom (25)")
		self.assertTrue(added)
		# Saved device is now an explicit choice, not at index 0.
		self.assertIn((25, "Custom (25)"), choices)
		self.assertEqual([idx for idx, _ in choices], [-1, 3, 25])

	def test_existing_device_is_not_duplicated(self):
		choices = [(-1, "System default"), (25, "USB Mic")]
		added = ensure_device_in_choices(choices, 25, label="Custom (25)")
		self.assertFalse(added)
		self.assertEqual(len(choices), 2)
		self.assertEqual([idx for idx, _ in choices], [-1, 25])

	def test_system_default_sentinel_is_skipped(self):
		# The system default (-1) is always the first entry; calling with
		# -1 must be a no-op so we don't add a duplicate "System default" row.
		choices = [(-1, "System default"), (5, "Webcam Mic")]
		added = ensure_device_in_choices(choices, -1, label="Should not be used")
		self.assertFalse(added)
		self.assertEqual(len(choices), 2)

	def test_negative_index_other_than_sentinel_is_skipped(self):
		# Defensive: any negative index is the "no device / use system" case
		# in this codebase, so we skip them.
		choices = [(-1, "System default")]
		added = ensure_device_in_choices(choices, -2, label="Bad")
		self.assertFalse(added)
		self.assertEqual(len(choices), 1)

	def test_preserves_order_of_existing_entries(self):
		choices = [(-1, "System default"), (3, "A"), (7, "B")]
		ensure_device_in_choices(choices, 25, label="C (25)")
		self.assertEqual([idx for idx, _ in choices], [-1, 3, 7, 25])

	def test_default_label_is_used_when_not_provided(self):
		choices = [(-1, "System default")]
		added = ensure_device_in_choices(choices, 25)
		self.assertTrue(added)
		_, label = choices[1]
		# The default label must mention the device index so the user
		# can identify which device is being preserved.
		self.assertIn("25", label)
		self.assertIn("not currently detected", label)

	def test_regression_index_for_device_no_longer_returns_0(self):
		"""The original bug: MicrophoneDialog._index_for_device returned 0
		for any device not in the list, silently mapping to "System default".
		With the fix, the device is in the list, so _index_for_device finds
		it at the right index.
		"""
		choices = [(-1, "System default"), (3, "A"), (7, "B")]
		ensure_device_in_choices(choices, 25, label="C (25)")
		# Simulate the dialog's lookup:
		target = 25
		index = next(
			(i for i, (idx, _) in enumerate(choices) if idx == target),
			0,
		)
		self.assertNotEqual(index, 0, "Saved device would have been silently mapped to 'System default'")
		self.assertEqual(index, 3)
		# And clicking OK would preserve the device, not overwrite it.
		self.assertEqual(choices[index][0], 25)


class AudioRecorderStopPortAudioTests(unittest.TestCase):
	"""Pin the AudioRecorder.stop() PortAudio error handling.

	The original bug: PortAudio's stop_stream() can raise OSError
	[Errno -9987] on some Windows drivers. The old code let that
	propagate out of stop() and leave the add-on stuck in "processing".
	The fix isolates the call so close() still runs and the caller
	always gets a wav path back.
	"""

	def setUp(self) -> None:
		from globalPlugins.groqVoiceDictation.audio_recorder import (  # noqa: WPS433
			AudioRecorder, AudioRecorderError,
		)
		self._AudioRecorderError = AudioRecorderError
		# Build a recorder that is in the "recording" state but with a
		# stream whose stop_stream() / close() will raise OSError.
		self.recorder = AudioRecorder(
			on_silence=None,
			input_device_index=-1,
		)
		self.recorder._recording = True

		class _Boom:
			def stop_stream(self):
				raise OSError(-9987, "Wait timed out")

			def close(self):
				raise OSError(-9999, "Stream is closed")

		self.recorder._stream = _Boom()
		self.recorder._pa = object()  # anything truthy

		# Stub the wav writer so we don't need a real PyAudio stream
		# to have produced audio.
		self._write_calls: list[str] = []

		def _fake_write() -> str:
			self._write_calls.append("wrote")
			return "C:/tmp/fake.wav"

		self.recorder._write_temp_wave = _fake_write  # type: ignore[assignment]

	def test_oserror_on_stop_stream_does_not_propagate(self):
		# Before the fix this raised OSError. After the fix it must
		# return the wav path normally.
		path = self.recorder.stop()
		self.assertEqual(path, "C:/tmp/fake.wav")
		# And the wav must have been written so the dictation isn't lost.
		self.assertEqual(self._write_calls, ["wrote"])

	def test_stream_and_pa_cleared_after_stop(self):
		self.recorder.stop()
		self.assertIsNone(self.recorder._stream)
		self.assertIsNone(self.recorder._pa)

	def test_subsequent_stop_raises_audio_recorder_error(self):
		# After stop(), the recorder is no longer "recording" — calling
		# stop() again must raise, not silently no-op.
		self.recorder.stop()
		with self.assertRaises(self._AudioRecorderError):
			self.recorder.stop()


class StopAndProcessBulletproofTests(unittest.TestCase):
	"""Pin the _stop_and_process contract: any exception during recorder
	stop() must release the _processing flag so the add-on is never left
	stuck. The original bug was that an OSError from PortAudio's
	stop_stream() escaped the function and left _processing = True with
	no worker to ever reset it.
	"""

	def setUp(self) -> None:
		GlobalPlugin = _plugin.GlobalPlugin
		self.plugin = object.__new__(GlobalPlugin)
		self.plugin._recorder = None
		self.plugin._processing = False
		self.plugin._state_lock = __import__("threading").Lock()
		self.plugin._pending_text = None
		self.plugin._confirm_timer = None
		self.plugin._confirm_gestures_bound = False
		self.plugin._preflight_timer = None
		self.plugin._last_toggle_time = 0.0
		self.plugin._double_press_window_ms = 500
		self.plugin._cancel_token = 0
		self.plugin._notify = mock.MagicMock()

	def test_oserror_during_stop_does_not_leave_processing_flag_set(self):
		recorder = mock.MagicMock()
		recorder.is_recording = True
		recorder.has_speech.return_value = True
		# The exact PortAudio symptom from the user's log.
		recorder.stop.side_effect = OSError(-9987, "Wait timed out")
		self.plugin._recorder = recorder
		self.plugin._cancel_preflight = mock.MagicMock()

		# This must NOT raise. Before the fix it raised OSError out of the
		# script handler and left _processing = True forever.
		self.plugin._stop_and_process()

		self.assertFalse(
			self.plugin._processing,
			"_processing must be reset so the add-on is not stuck",
		)
		# And the user got told something went wrong, not silence.
		self.plugin._notify.assert_called()


if __name__ == "__main__":
	unittest.main()

