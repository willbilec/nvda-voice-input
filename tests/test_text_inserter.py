"""Tests for the text insertion path.

These tests pin the order of operations in ``_paste_text`` — the
backup-before-copy contract that prevents the last dictation from
lingering on the user's clipboard after a paste-fallback insertion.

The original bug: ``_paste_text`` called ``api.copyToClip(text)``
first, then read the clipboard with ``api.getClipData()`` to capture
the "backup". By that point the clipboard already held the dictation
text, so the backup was the dictation text itself. The deferred
``_restore_clipboard`` then put the dictation text back on the
clipboard instead of the user's original content. Symptom: the last
dictation stayed on the clipboard after a paste-fallback insertion,
even after the user had toggled off paste fallback in the Debugging
settings panel.

NVDA modules (api, core, wx, keyboardHandler, watchdog, logHandler,
addonHandler, etc.) are stubbed out so the add-on package — and
therefore text_inserter — can be imported in isolation.
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
TEXT_INSERTER_PKG = "globalPlugins.groqVoiceDictation.text_inserter"


def _install_nvda_stubs() -> None:
	"""Install every NVDA-module stub needed to import the add-on.

	The add-on's ``__init__.py`` imports addonHandler, gui, wx, etc.
	at module top level. Every one of those has to be in sys.modules
	before the import, but only as a shape — the tests patch every
	NVDA call site individually.
	"""
	if "addonHandler" not in sys.modules:
		_stub = types.ModuleType("addonHandler")
		_stub.AddonError = type("AddonError", (Exception,), {})

		def _stub_init_translation():
			# NVDA's real initTranslation installs the gettext _
			# function into builtins so module-level code can call
			# _("string"). In tests we don't need translation, so
			# just identity-wrap.
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
		_stub = types.ModuleType("tones")
		_stub.beep = lambda *args, **kwargs: None
		sys.modules["tones"] = _stub

	if "ui" not in sys.modules:
		_stub = types.ModuleType("ui")
		_stub.message = lambda *args, **kwargs: None
		sys.modules["ui"] = _stub

	if "wx" not in sys.modules:
		_stub = types.ModuleType("wx")
		_WX_BASE = type("_WX_BASE", (), {})
		for _name in (
			"Window", "Dialog", "Panel", "Choice", "CheckBox", "Button",
			"StaticText", "BoxSizer", "TextCtrl", "MessageDialog", "MessageBox",
			"Size", "Sizer", "Font", "Colour",
		):
			setattr(_stub, _name, _WX_BASE)
		_stub.ID_OK = 1
		_stub.ID_YES = 2
		_stub.OK = 0x4000
		_stub.LEFT = 0
		_stub.EXPAND = 0
		_stub.ALIGN_CENTER_HORIZONTAL = 0
		_stub.TE_MULTILINE = 0
		_stub.EVT_BUTTON = "EVT_BUTTON"
		_stub.EVT_CHOICE = "EVT_CHOICE"
		_stub.CallLater = type(
			"CallLater", (),
			{"__init__": lambda self, *a, **k: None, "Stop": lambda self: None},
		)
		_stub.CallAfter = lambda fn, *a, **k: fn(*a)
		sys.modules["wx"] = _stub

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
		_stub = types.ModuleType("scriptHandler")
		_stub.script = lambda *dargs, **dkwargs: (lambda fn: fn)
		sys.modules["scriptHandler"] = _stub

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
	# pyaudio is a C extension; we stub it so audio_recorder can
	# import without the real _portaudio module being present.
	if "pyaudio" not in sys.modules:
		sys.modules["pyaudio"] = types.SimpleNamespace(
			PyAudio=type("PyAudio", (), {}),
			paInt16=8,
			paContinue=0,
		)

	if str(MODULE_DIR) not in sys.path:
		sys.path.insert(0, str(MODULE_DIR))
	if str(MODULE_DIR / "lib") not in sys.path:
		sys.path.insert(0, str(MODULE_DIR / "lib"))


def _load_text_inserter():
	"""Import (or reload) the text_inserter module with stubs in place.

	The package ``__init__.py`` does ``from .text_inserter import
	TextInserter`` at module load, so we have to import the package
	first. To keep the import cheap we stub the settings_panel —
	its dialog builders pull in wx widgets we don't need here.
	"""
	_install_nvda_stubs()
	_sp_stub = types.ModuleType("globalPlugins.groqVoiceDictation.settings_panel")
	_sp_stub.GroqVoiceDictationSettingsPanel = type(
		"GroqVoiceDictationSettingsPanel", (), {}
	)
	sys.modules["globalPlugins.groqVoiceDictation.settings_panel"] = _sp_stub
	for mod_name in (TEXT_INSERTER_PKG, PLUGIN_PKG, PACKAGE_PARENT.name):
		sys.modules.pop(mod_name, None)
	import globalPlugins  # noqa: F401  -- namespace package
	import globalPlugins.groqVoiceDictation  # noqa: F401
	return importlib.reload(sys.modules[TEXT_INSERTER_PKG])


_install_nvda_stubs()
_TEXT_INSERTER_MODULE = _load_text_inserter()
TextInserter = _TEXT_INSERTER_MODULE.TextInserter


def _make_focus(window_class: str = "Edit", handle: int = 0):
	"""Build a minimal focus object with the attributes _paste_text reads."""
	focus = mock.MagicMock()
	focus.windowClassName = window_class
	focus.windowHandle = handle
	return focus


def _patch_api(name, **kwargs):
	"""Patch an ``api`` attribute on the text_inserter module.

	``mock.patch.object`` requires the attribute to already exist
	on the target, but our ``api`` stub starts empty (the real
	NVDA ``api`` module is full of functions we don't need to model
	here). This wrapper adds ``create=True`` so each test can patch
	any subset of ``copyToClip`` / ``getClipData`` /
	``processPendingEvents`` / ``getFocusObject`` without
	pre-seeding the module.
	"""
	return mock.patch.object(_TEXT_INSERTER_MODULE.api, name, create=True, **kwargs)


def _patch_gesture_from_name(**kwargs):
	"""Patch ``KeyboardInputGesture.fromName`` with ``create=True``.

	The ``KeyboardInputGesture`` stub is a bare class — ``fromName``
	is only present on the real NVDA class. The text_inserter code
	only ever calls ``fromName(...).send()``, so creating the
	attribute on the class is the cleanest way to patch it.
	"""
	return mock.patch.object(
		_TEXT_INSERTER_MODULE.KeyboardInputGesture, "fromName",
		create=True, **kwargs,
	)


class UnicodeTypingLatencyTests(unittest.TestCase):
	def test_long_text_uses_large_batches_without_fixed_sleeps(self):
		inserter = TextInserter()
		send_input = mock.MagicMock(side_effect=lambda count, *_args: count)
		fake_windll = types.SimpleNamespace(
			user32=types.SimpleNamespace(SendInput=send_input)
		)
		with mock.patch.object(_TEXT_INSERTER_MODULE.ctypes, "windll", fake_windll, create=True), \
				mock.patch.object(_TEXT_INSERTER_MODULE.time, "sleep") as sleep_mock:
			self.assertTrue(inserter._type_unicode("a" * 600))
		# 600 UTF-16 units at 256 units/batch = three Win32 calls.
		self.assertEqual(send_input.call_count, 3)
		sleep_mock.assert_not_called()


class PasteTextBackupOrderTests(unittest.TestCase):
	"""Pin the backup-before-copy order in ``_paste_text``.

	These are the regression tests for the bug where the dictation
	text stayed on the clipboard after a paste-fallback insertion.
	"""

	def test_clipboard_is_backed_up_before_copy(self):
		"""``api.getClipData`` must be called BEFORE ``api.copyToClip``.

		The old order (copy first, then read) captured the dictation
		text as the "backup", so the deferred restore put the
		dictation text back on the clipboard. Pin the order so the
		regression cannot be reintroduced.
		"""
		inserter = TextInserter()
		focus = _make_focus(window_class="Edit")
		# Use a single parent mock as a recorder. Both getClipData
		# and copyToClip record a marker on it, in call order, so
		# we can assert getClipData fired first.
		recorder = mock.MagicMock()
		with _patch_api(
			"getClipData",
			side_effect=lambda: (recorder.mark("get"), "user-clipboard")[1],
		), _patch_api(
			"copyToClip",
			side_effect=lambda t: (recorder.mark("copy"), True)[1],
		), _patch_api(
			"processPendingEvents", return_value=None,
		), _patch_gesture_from_name() as gesture_factory:
			gesture_factory.return_value.send = mock.MagicMock()
			inserter._paste_text("dictation text", focus)
		self.assertEqual(
			recorder.mock_calls,
			[mock.call.mark("get"), mock.call.mark("copy")],
		)

	def test_restore_receives_original_clipboard_not_dictation(self):
		"""The deferred ``_restore_clipboard`` must be called with the
		user's original clipboard content, not the dictation text.

		The old bug: the backup was taken after the copy, so it was
		the dictation text. The restore then put the dictation text
		back on the clipboard.

		This test models the real clipboard semantics: ``getClipData``
		returns whatever is currently on the clipboard. If the
		add-on has just called ``copyToClip(text)``, the clipboard
		holds the dictation text, not the user's original content.
		With the bug, the backup would be the dictation text; with
		the fix, it is the user's original content.
		"""
		inserter = TextInserter()
		focus = _make_focus(window_class="Edit")
		clipboard_state = {"contents": "ORIGINAL-CLIPBOARD"}

		def fake_copy(text):
			# Real ``api.copyToClip`` overwrites the clipboard with
			# the given text.
			clipboard_state["contents"] = text
			return True

		def fake_get():
			# Real ``api.getClipData`` returns whatever is on the
			# clipboard right now.
			return clipboard_state["contents"]

		with _patch_api("getClipData", side_effect=fake_get), \
				_patch_api("copyToClip", side_effect=fake_copy), \
				_patch_api("processPendingEvents", return_value=None), \
				_patch_gesture_from_name() as gesture_factory, \
				mock.patch.object(_TEXT_INSERTER_MODULE.threading, "Thread") as thread_factory, \
				mock.patch.object(_TEXT_INSERTER_MODULE.time, "sleep", return_value=None):
			gesture_factory.return_value.send = mock.MagicMock()
			thread_instance = mock.MagicMock()
			thread_factory.return_value = thread_instance
			inserter._paste_text("DICTATION-TEXT", focus)
		# A Thread was started with _restore_clipboard as the target.
		thread_factory.assert_called_once()
		self.assertEqual(
			thread_factory.call_args.kwargs["target"],
			inserter._restore_clipboard,
		)
		# The args must be the ORIGINAL clipboard content, not the
		# dictation text. With the bug, ``getClipData`` is called
		# after ``copyToClip`` and returns "DICTATION-TEXT".
		self.assertEqual(
			thread_factory.call_args.kwargs["args"],
			("ORIGINAL-CLIPBOARD",),
		)

	def test_restore_skipped_when_getClipData_raises_oserror(self):
		"""If reading the clipboard raises OSError, fall back to
		leaving the dictation text on the clipboard and skip the
		restore (the current behavior, preserved for safety).
		"""
		inserter = TextInserter()
		focus = _make_focus(window_class="Edit")
		with _patch_api("getClipData", side_effect=OSError("clipboard locked")), \
				_patch_api("copyToClip", return_value=True), \
				_patch_api("processPendingEvents", return_value=None), \
				_patch_gesture_from_name() as gesture_factory, \
				mock.patch.object(_TEXT_INSERTER_MODULE.threading, "Thread") as thread_factory:
			gesture_factory.return_value.send = mock.MagicMock()
			result = inserter._paste_text("dictation text", focus)
		self.assertTrue(result)
		thread_factory.assert_not_called()

	def test_copy_failure_returns_false_and_does_not_paste(self):
		"""If ``api.copyToClip`` fails 3 times, ``_paste_text`` must
		return False and not send the paste gesture.
		"""
		inserter = TextInserter()
		focus = _make_focus(window_class="Edit")
		with _patch_api("getClipData", return_value="orig"), \
				_patch_api("copyToClip", return_value=False), \
				_patch_api("processPendingEvents", return_value=None), \
				_patch_gesture_from_name() as gesture_factory, \
				mock.patch.object(_TEXT_INSERTER_MODULE.time, "sleep", return_value=None):
			gesture_factory.return_value.send = mock.MagicMock()
			result = inserter._paste_text("dictation text", focus)
		self.assertFalse(result)
		gesture_factory.assert_not_called()

	def test_normal_focus_sends_control_v(self):
		"""A non-Console focus should use the Ctrl+V keyboard gesture."""
		inserter = TextInserter()
		focus = _make_focus(window_class="Edit")
		with _patch_api("getClipData", return_value="orig"), \
				_patch_api("copyToClip", return_value=True), \
				_patch_api("processPendingEvents", return_value=None), \
				_patch_gesture_from_name() as gesture_factory, \
				mock.patch.object(_TEXT_INSERTER_MODULE.threading, "Thread"):
			send = mock.MagicMock()
			gesture_factory.return_value.send = send
			inserter._paste_text("text", focus)
		gesture_factory.assert_called_once_with("control+v")
		send.assert_called_once()

	def test_successful_paste_has_no_fixed_foreground_sleep(self):
		inserter = TextInserter()
		focus = _make_focus(window_class="Edit")
		with _patch_api("getClipData", return_value="orig"), \
				_patch_api("copyToClip", return_value=True), \
				_patch_api("processPendingEvents", return_value=None), \
				_patch_gesture_from_name() as gesture_factory, \
				mock.patch.object(_TEXT_INSERTER_MODULE.threading, "Thread"), \
				mock.patch.object(_TEXT_INSERTER_MODULE.time, "sleep") as sleep_mock:
			gesture_factory.return_value.send = mock.MagicMock()
			self.assertTrue(inserter._paste_text("text", focus))
		sleep_mock.assert_not_called()

	def test_console_focus_uses_watchdog_paste_message(self):
		"""A Console focus should use the WM_COMMAND paste path,
		not the Ctrl+V gesture.
		"""
		inserter = TextInserter()
		focus = _make_focus(window_class="ConsoleWindowClass", handle=12345)
		with _patch_api("getClipData", return_value="orig"), \
				_patch_api("copyToClip", return_value=True), \
				_patch_api("processPendingEvents", return_value=None), \
				_patch_gesture_from_name() as gesture_factory, \
				mock.patch.object(_TEXT_INSERTER_MODULE.threading, "Thread"), \
				mock.patch.object(_TEXT_INSERTER_MODULE.watchdog, "cancellableSendMessage", create=True) as send_msg:
			gesture_factory.return_value.send = mock.MagicMock()
			inserter._paste_text("text", focus)
		gesture_factory.assert_not_called()
		send_msg.assert_called_once_with(
			12345, _TEXT_INSERTER_MODULE.WM_COMMAND, _TEXT_INSERTER_MODULE.CONSOLE_PASTE, 0,
		)


class InsertRoutingTests(unittest.TestCase):
	"""Pin the routing in ``insert`` — typing first, paste as fallback."""

	def setUp(self) -> None:
		self.inserter = TextInserter()
		self.focus = _make_focus(window_class="Edit")

	def test_typing_success_skips_paste(self):
		"""When ``_type_unicode`` returns True, ``_paste_text`` must
		not be called and the clipboard must not be touched.
		"""
		with mock.patch.object(self.inserter, "_type_unicode", return_value=True) as type_unicode, \
				mock.patch.object(self.inserter, "_paste_text", return_value=True) as paste_text, \
				_patch_api("getFocusObject", return_value=self.focus):
			result = self.inserter.insert("hello", allow_paste_fallback=True)
		self.assertTrue(result)
		type_unicode.assert_called_once_with("hello")
		paste_text.assert_not_called()

	def test_typing_failure_falls_back_to_paste_when_allowed(self):
		"""When ``_type_unicode`` returns False and paste fallback is
		allowed, ``_paste_text`` must be called.
		"""
		with mock.patch.object(self.inserter, "_type_unicode", return_value=False), \
				mock.patch.object(self.inserter, "_paste_text", return_value=True) as paste_text, \
				_patch_api("getFocusObject", return_value=self.focus):
			result = self.inserter.insert("hello", allow_paste_fallback=True)
		self.assertTrue(result)
		paste_text.assert_called_once_with("hello", self.focus)

	def test_typing_exception_falls_back_to_paste_when_allowed(self):
		"""When ``_type_unicode`` raises, the exception is swallowed
		and ``_paste_text`` is called (when allowed).
		"""
		with mock.patch.object(self.inserter, "_type_unicode", side_effect=RuntimeError("SendInput crash")), \
				mock.patch.object(self.inserter, "_paste_text", return_value=True) as paste_text, \
				_patch_api("getFocusObject", return_value=self.focus):
			result = self.inserter.insert("hello", allow_paste_fallback=True)
		self.assertTrue(result)
		paste_text.assert_called_once_with("hello", self.focus)

	def test_typing_failure_with_paste_disabled_returns_false(self):
		"""When ``_type_unicode`` returns False and paste fallback is
		disabled, ``_paste_text`` must NOT be called and ``insert``
		must return False. The clipboard must not be touched.
		"""
		with mock.patch.object(self.inserter, "_type_unicode", return_value=False), \
				mock.patch.object(self.inserter, "_paste_text") as paste_text, \
				_patch_api("getFocusObject", return_value=self.focus), \
				_patch_api("getClipData") as get_clip, \
				_patch_api("copyToClip") as copy_clip:
			result = self.inserter.insert("hello", allow_paste_fallback=False)
		self.assertFalse(result)
		paste_text.assert_not_called()
		get_clip.assert_not_called()
		copy_clip.assert_not_called()

	def test_console_focus_skips_typing_uses_paste_when_allowed(self):
		"""A Console focus must skip the typing path and go straight
		to ``_paste_text`` (which routes through the WM_COMMAND path).
		"""
		console_focus = _make_focus(window_class="ConsoleWindowClass", handle=99)
		with mock.patch.object(self.inserter, "_type_unicode") as type_unicode, \
				mock.patch.object(self.inserter, "_paste_text", return_value=True) as paste_text, \
				_patch_api("getFocusObject", return_value=console_focus):
			result = self.inserter.insert("hello", allow_paste_fallback=True)
		self.assertTrue(result)
		type_unicode.assert_not_called()
		paste_text.assert_called_once_with("hello", console_focus)

	def test_console_focus_with_paste_disabled_returns_false(self):
		"""A Console focus with paste fallback disabled cannot be
		inserted into at all (typing is skipped for console, paste
		is disabled). The function must return False and not touch
		the clipboard.
		"""
		console_focus = _make_focus(window_class="ConsoleWindowClass", handle=99)
		with mock.patch.object(self.inserter, "_type_unicode") as type_unicode, \
				mock.patch.object(self.inserter, "_paste_text") as paste_text, \
				_patch_api("getFocusObject", return_value=console_focus), \
				_patch_api("getClipData") as get_clip, \
				_patch_api("copyToClip") as copy_clip:
			result = self.inserter.insert("hello", allow_paste_fallback=False)
		self.assertFalse(result)
		type_unicode.assert_not_called()
		paste_text.assert_not_called()
		get_clip.assert_not_called()
		copy_clip.assert_not_called()

	def test_empty_text_returns_false(self):
		"""An empty text must short-circuit and not touch anything."""
		with mock.patch.object(self.inserter, "_type_unicode") as type_unicode, \
				mock.patch.object(self.inserter, "_paste_text") as paste_text:
			result = self.inserter.insert("", allow_paste_fallback=True)
		self.assertFalse(result)
		type_unicode.assert_not_called()
		paste_text.assert_not_called()


if __name__ == "__main__":
	unittest.main()
