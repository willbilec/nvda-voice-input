import array
import ctypes
import threading
import time

import api
import core
from keyboardHandler import KeyboardInputGesture
import watchdog


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
WM_COMMAND = 0x0111
CONSOLE_PASTE = 0xFFF1


class KEYBDINPUT(ctypes.Structure):
	_fields_ = [
		("wVk", ctypes.c_ushort),
		("wScan", ctypes.c_ushort),
		("dwFlags", ctypes.c_ulong),
		("time", ctypes.c_ulong),
		("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
	]


class _INPUTUNION(ctypes.Union):
	_fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
	_fields_ = [("type", ctypes.c_ulong), ("union", _INPUTUNION)]


class TextInserter:
	def insert(self, text: str, allow_paste_fallback: bool = True) -> bool:
		if not text:
			return False
		focus = api.getFocusObject()
		if focus and getattr(focus, "windowClassName", "") != "ConsoleWindowClass":
			try:
				if self._type_unicode(text):
					return True
			except Exception:
				pass
		if allow_paste_fallback:
			return self._paste_text(text, focus)
		return False

	def _type_unicode(self, text: str) -> bool:
		utf16_units = array.array("H")
		utf16_units.frombytes(text.encode("utf-16-le"))
		inputs = []
		for unit in utf16_units:
			inputs.append(INPUT(type=INPUT_KEYBOARD, union=_INPUTUNION(ki=KEYBDINPUT(0, unit, KEYEVENTF_UNICODE, 0, None))))
			inputs.append(INPUT(type=INPUT_KEYBOARD, union=_INPUTUNION(ki=KEYBDINPUT(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, None))))
		if not inputs:
			return False
		count = ctypes.windll.user32.SendInput(len(inputs), (INPUT * len(inputs))(*inputs), ctypes.sizeof(INPUT))
		return count == len(inputs)

	def _paste_text(self, text: str, focus) -> bool:
		try:
			clipboard_backup = api.getClipData()
		except OSError:
			clipboard_backup = None
		if not api.copyToClip(text):
			return False
		time.sleep(0.05)
		api.processPendingEvents(False)
		if focus and getattr(focus, "windowClassName", "") == "ConsoleWindowClass":
			watchdog.cancellableSendMessage(focus.windowHandle, WM_COMMAND, CONSOLE_PASTE, 0)
		else:
			KeyboardInputGesture.fromName("control+v").send()
		if clipboard_backup is not None:
			threading.Thread(target=self._restore_clipboard, args=(clipboard_backup,), daemon=True).start()
		return True

	def _restore_clipboard(self, backup_text: str) -> None:
		time.sleep(0.3)
		core.callLater(0, api.copyToClip, backup_text)
