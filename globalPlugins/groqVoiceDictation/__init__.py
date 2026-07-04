# -*- coding: utf-8 -*-
import os
import sys
import threading

module_path = os.path.abspath(os.path.dirname(__file__))
lib_path = os.path.join(module_path, "lib")
if lib_path not in sys.path:
	sys.path.insert(0, lib_path)

import addonHandler
import globalPluginHandler
import gui
import tones
import ui
import wx
from logHandler import log
from scriptHandler import script

from . import config_manager
from .audio_recorder import AudioRecorder, AudioRecorderError
from .groq_client import GroqClient, GroqClientError, is_hallucination
from .gemini_client import GeminiClient, GeminiClientError
from .settings_panel import GroqVoiceDictationSettingsPanel
from .text_inserter import TextInserter

try:
	addonHandler.initTranslation()
except addonHandler.AddonError:
	log.warning("Unable to init translations. This may be because the add-on is running from scratchpad.")


_addon = addonHandler.getCodeAddon()
ADDON_SUMMARY = _addon.manifest["summary"]


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
	scriptCategory = ADDON_SUMMARY

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		config_manager.ensure_config_spec()
		gui.settingsDialogs.NVDASettingsDialog.categoryClasses.append(GroqVoiceDictationSettingsPanel)
		self._recorder = None
		self._processing = False
		self._text_inserter = TextInserter()
		self._state_lock = threading.Lock()
		self._pending_text: str | None = None
		self._confirm_timer: wx.CallLater | None = None
		self._confirm_gestures_bound: bool = False
		self._preflight_timer: wx.CallLater | None = None

	def terminate(self):
		self._cancel_preflight()
		self._clear_confirm_gestures()
		self._pending_text = None
		with self._state_lock:
			recorder = self._recorder
			self._recorder = None
		if recorder is not None and recorder.is_recording:
			try:
				wav_path = recorder.stop()
			except Exception:
				wav_path = None
			if wav_path:
				AudioRecorder.delete_file(wav_path)
		try:
			gui.settingsDialogs.NVDASettingsDialog.categoryClasses.remove(GroqVoiceDictationSettingsPanel)
		except ValueError:
			log.warning("Settings panel already removed for %s", ADDON_SUMMARY)
		super().terminate()

	@script(
		description=_("Toggle Groq voice dictation"),
		gesture="kb:NVDA+shift+v",
	)
	def script_toggleVoiceDictation(self, gesture):
		with self._state_lock:
			if self._processing:
				self._notify(_("Still processing the previous dictation."), tone=320, is_error=True)
				return
			if self._recorder is not None and self._recorder.is_recording:
				recorder = self._recorder
			else:
				recorder = None
		if recorder is not None:
			self._stop_and_process()
		else:
			self._start_recording()

	def _start_recording(self) -> None:
		conf = config_manager.get()
		if not conf["apiKey"].strip():
			self._notify(_("Set a Groq API key in the Groq Voice Dictation settings first."), tone=220, is_error=True)
			return
		fallback_enabled = conf["fallbackEnabled"]
		recorder = AudioRecorder(
			on_silence=self._handle_silence_stop,
			input_device_index=int(conf["microphoneDevice"]),
			silence_enabled=conf["silenceDetection"],
			silence_timeout=conf["silenceTimeout"],
			silence_threshold=conf["silenceThreshold"],
			fallback_device_index=int(conf["fallbackMicrophoneDevice"]) if fallback_enabled else -1,
		)
		try:
			recorder.start()
		except AudioRecorderError as exc:
			self._notify(str(exc), tone=220, is_error=True)
			return
		with self._state_lock:
			self._recorder = recorder
		if recorder.used_fallback:
			self._notify(_("Listening (using fallback microphone.)"), tone=880)
		else:
			if fallback_enabled and int(conf["fallbackMicrophoneDevice"]) != int(conf["microphoneDevice"]):
				delay = int(conf["fallbackPreflightMs"])
				self._preflight_timer = wx.CallLater(delay, self._check_preflight_silence, recorder)
			self._notify(_("Listening."), tone=880)

	def _check_preflight_silence(self, recorder: AudioRecorder) -> None:
		self._preflight_timer = None
		with self._state_lock:
			if self._recorder is not recorder or not recorder.is_recording:
				return
		if recorder.has_speech():
			return
		with self._state_lock:
			if self._recorder is not recorder:
				return
			self._recorder = None
		try:
			wav_path = recorder.stop()
		except Exception:
			wav_path = None
		if wav_path:
			AudioRecorder.delete_file(wav_path)
		self._notify(_("Primary microphone silent. Switching to fallback."), tone=520)
		self._start_fallback_recording()

	def _cancel_preflight(self) -> None:
		if self._preflight_timer is not None:
			self._preflight_timer.Stop()
			self._preflight_timer = None

	def _start_fallback_recording(self) -> None:
		conf = config_manager.get()
		recorder = AudioRecorder(
			on_silence=self._handle_silence_stop,
			input_device_index=int(conf["fallbackMicrophoneDevice"]),
			silence_enabled=conf["silenceDetection"],
			silence_timeout=conf["silenceTimeout"],
			silence_threshold=conf["silenceThreshold"],
			fallback_device_index=-1,
		)
		try:
			recorder.start()
		except AudioRecorderError as exc:
			self._notify(str(exc), tone=220, is_error=True)
			return
		with self._state_lock:
			self._recorder = recorder
		self._notify(_("Listening (using fallback microphone.)"), tone=880)

	def _handle_silence_stop(self) -> None:
		self._cancel_preflight()
		with self._state_lock:
			if self._recorder is None or not self._recorder.is_recording or self._processing:
				return
		self._stop_and_process(auto_stop=True)

	def _stop_and_process(self, auto_stop: bool = False) -> None:
		self._cancel_preflight()
		with self._state_lock:
			recorder = self._recorder
			if recorder is None or not recorder.is_recording:
				return
			self._recorder = None
			self._processing = True
		try:
			has_speech = recorder.has_speech()
			wav_path = recorder.stop()
		except AudioRecorderError as exc:
			with self._state_lock:
				self._processing = False
			self._notify(str(exc), tone=220, is_error=True)
			return
		stop_message = _("Silence detected. Processing.") if auto_stop else _("Stopped listening. Processing.")
		self._notify(stop_message, tone=660)
		threading.Thread(target=self._process_recording, args=(wav_path, has_speech, recorder.used_fallback), daemon=True).start()

	def _process_recording(self, wav_path: str, has_speech: bool, used_fallback: bool) -> None:
		conf = config_manager.get()
		transcription_client = GroqClient(api_key=conf["apiKey"])
		transcription_model = conf["transcriptionModel"]
		transcription_prompt = config_manager.get_active_prompt(conf)
		_confirm_pending = False
		try:
			if not has_speech:
				if not used_fallback and conf["fallbackEnabled"]:
					fallback_dev = int(conf.get("fallbackMicrophoneDevice", -1))
					primary_dev = int(conf.get("microphoneDevice", -1))
					if fallback_dev != primary_dev:
						AudioRecorder.delete_file(wav_path)
						wav_path = ""
						self._notify(_("Primary microphone silent. Trying fallback."), tone=520)
						with self._state_lock:
							self._processing = False
						wx.CallAfter(self._start_fallback_recording)
						return
				self._notify(_("No speech was detected."), tone=260, is_error=True)
				return
			self._notify(_("Transcribing."), tone=520)
			transcript = transcription_client.transcribe(
				wav_path,
				transcription_model,
				prompt=transcription_prompt,
				language=conf.get("transcriptionLanguage", "en"),
			)
			if not transcript.strip() or is_hallucination(transcript):
				self._notify(_("No speech was detected."), tone=260, is_error=True)
				return
			log.info("Raw transcript: %r", transcript)
			if conf["speakRawTranscript"]:
				wx.CallAfter(ui.message, _("Raw transcript: %s") % transcript)
			final_text = transcript
			if conf["cleanupMode"] != "raw":
				cleanup_model = conf["cleanupModel"]
				is_gemini = cleanup_model.startswith("gemini") if cleanup_model else False
				cleanup_client: GeminiClient | None = None
				if is_gemini:
					gk = conf.get("geminiApiKey", "").strip()
					if gk:
						cleanup_client = GeminiClient(api_key=gk)
				else:
					cleanup_client = GroqClient(api_key=conf["apiKey"])
				if cleanup_client is None:
					self._notify(_("Gemini key not set. Inserting raw transcript."), tone=420, is_error=True)
					final_text = transcript
				else:
					try:
						final_text = cleanup_client.cleanup(transcript, conf["cleanupMode"], cleanup_model)
					except GroqClientError as exc:
						log.error("Groq cleanup failed: %s (%s)", exc.message, exc.category)
						self._notify(_("Cleanup failed. Inserting the raw transcript."), tone=420, is_error=True)
						final_text = transcript
					except GeminiClientError as exc:
						log.error("Gemini cleanup failed: %s (%s)", exc.message, exc.category)
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
		except (GroqClientError, GeminiClientError) as exc:
			log.error("Dictation failed: %s (%s)", exc.message, exc.category)
			self._notify(exc.message, tone=220, is_error=True)
		except Exception:
			log.exception("Unexpected Groq Voice Dictation failure")
			self._notify(_("Unexpected dictation error. Check the NVDA log for details."), tone=220, is_error=True)
		finally:
			AudioRecorder.delete_file(wav_path)
			if not _confirm_pending:
				with self._state_lock:
					self._processing = False

	def _clear_confirm_gestures(self) -> None:
		if not self._confirm_gestures_bound:
			return
		self._confirm_gestures_bound = False
		self.removeGestureBinding("kb:space")
		self.removeGestureBinding("kb:escape")
		if self._confirm_timer is not None:
			self._confirm_timer.Stop()
			self._confirm_timer = None

	def _start_confirm_window(self, text: str) -> None:
		conf = config_manager.get()
		self._pending_text = text
		ui.message(text)
		self.bindGesture("kb:space", "cancelPendingDictation")
		self.bindGesture("kb:escape", "cancelPendingDictation")
		self._confirm_gestures_bound = True
		self._confirm_timer = wx.CallLater(conf["confirmTimeout"] * 1000, self._execute_pending_insert)

	def script_cancelPendingDictation(self, gesture) -> None:
		self._clear_confirm_gestures()
		self._pending_text = None
		with self._state_lock:
			self._processing = False
		self._notify(_("Dictation cancelled."))

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

	def _notify(self, message: str, tone: int = 440, is_error: bool = False) -> None:
		mode = config_manager.get()["feedbackMode"]
		if mode in ("speech", "both"):
			wx.CallAfter(ui.message, message)
		if mode in ("tones", "both"):
			duration = 180 if is_error else 120
			wx.CallAfter(tones.beep, tone, duration)
