from typing import Any
import threading
import webbrowser

import addonHandler
import config
from gui import guiHelper, nvdaControls
from gui.settingsDialogs import SettingsPanel
from logHandler import log
import ui
import wx

from .audio_recorder import AudioRecorder, calculate_peak_level, list_input_devices
from . import config_manager

try:
	addonHandler.initTranslation()
except addonHandler.AddonError:
	log.warning("Unable to init translations in settings panel.")


_addon = addonHandler.getCodeAddon()
ADDON_SUMMARY = _addon.manifest["summary"]


class GroqVoiceDictationSettingsPanel(SettingsPanel):
	title = ADDON_SUMMARY

	def makeSettings(self, settingsSizer: wx.Sizer) -> None:
		sizer_helper = guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
		conf = config_manager.get()
		self._microphone_choices = [(-1, _("System default microphone"))]
		try:
			self._microphone_choices.extend(list_input_devices())
		except Exception:
			log.exception("Could not list microphone devices")

		self.api_key = sizer_helper.addLabeledControl(
			_("Groq API &key:"),
			wx.TextCtrl,
			value=conf["apiKey"],
			style=wx.TE_PASSWORD,
		)
		self.get_api_key_button = sizer_helper.addItem(
			wx.Button(self, label=_("Get API key"))
		)

		self.microphone_device = sizer_helper.addLabeledControl(
			_("&Microphone:"),
			wx.Choice,
			choices=[label for _value, label in self._microphone_choices],
		)
		self.microphone_device.SetSelection(
			self._selection_for_microphone_device(int(conf["microphoneDevice"]))
		)

		self.transcription_model = sizer_helper.addLabeledControl(
			_("&Transcription model:"),
			wx.Choice,
			choices=config_manager.TRANSCRIPTION_MODELS,
		)
		self.transcription_model.SetSelection(
			config_manager.TRANSCRIPTION_MODELS.index(conf["transcriptionModel"])
			if conf["transcriptionModel"] in config_manager.TRANSCRIPTION_MODELS
			else 0
		)

		self.cleanup_mode = sizer_helper.addLabeledControl(
			_("&Cleanup mode:"),
			wx.Choice,
			choices=config_manager.label_list(config_manager.CLEANUP_MODES),
		)
		self.cleanup_mode.SetSelection(
			config_manager.index_for_value(config_manager.CLEANUP_MODES, conf["cleanupMode"])
		)

		self.cleanup_model = sizer_helper.addLabeledControl(
			_("C&leanup model:"),
			wx.Choice,
			choices=config_manager.CLEANUP_MODELS,
		)
		self.cleanup_model.SetSelection(
			config_manager.CLEANUP_MODELS.index(conf["cleanupModel"])
			if conf["cleanupModel"] in config_manager.CLEANUP_MODELS
			else 0
		)

		self.silence_detection = sizer_helper.addItem(
			wx.CheckBox(self, label=_("Enable &silence detection"))
		)
		self.silence_detection.SetValue(conf["silenceDetection"])

		self.silence_timeout = sizer_helper.addLabeledControl(
			_("Silence timeout (&seconds):"),
			nvdaControls.SelectOnFocusSpinCtrl,
			value=str(conf["silenceTimeout"]),
			min=1,
			max=15,
		)

		self.feedback_mode = sizer_helper.addLabeledControl(
			_("&Feedback mode:"),
			wx.Choice,
			choices=config_manager.label_list(config_manager.FEEDBACK_MODES),
		)
		self.feedback_mode.SetSelection(
			config_manager.index_for_value(config_manager.FEEDBACK_MODES, conf["feedbackMode"])
		)

		self.allow_paste_fallback = sizer_helper.addItem(
			wx.CheckBox(self, label=_("Allow &paste fallback when typing fails"))
		)
		self.allow_paste_fallback.SetValue(conf["allowPasteFallback"])

		self.silence_threshold = sizer_helper.addLabeledControl(
			_("Silence &threshold:"),
			nvdaControls.SelectOnFocusSpinCtrl,
			value=str(conf["silenceThreshold"]),
			min=100,
			max=32767,
		)

		self.sample_mic_button = sizer_helper.addItem(
			wx.Button(self, label=_("Sa&mple microphone level"))
		)

		self.Bind(wx.EVT_BUTTON, self.on_get_api_key, self.get_api_key_button)
		self.Bind(wx.EVT_BUTTON, self.on_sample_mic, self.sample_mic_button)
		self.addon_help_note = sizer_helper.addItem(
			wx.StaticText(
				self,
				label=_("Use NVDA's installed add-ons dialog for this add-on's documentation."),
			)
		)

	def postInit(self) -> None:
		self.api_key.SetFocus()

	def _selection_for_microphone_device(self, saved_index: int) -> int:
		for selection, (device_index, _label) in enumerate(self._microphone_choices):
			if device_index == saved_index:
				return selection
		return 0

	def on_sample_mic(self, _event) -> None:
		self.sample_mic_button.Disable()
		self.sample_mic_button.SetLabel(_("Sampling..."))
		device_index = self._microphone_choices[self.microphone_device.GetSelection()][0]
		threading.Thread(target=self._do_sample_mic, args=(device_index,), daemon=True).start()

	def _do_sample_mic(self, device_index: int) -> None:
		import pyaudio
		peak = 0
		try:
			pa = pyaudio.PyAudio()
			try:
				stream = pa.open(
					format=pyaudio.paInt16,
					channels=AudioRecorder.channels,
					rate=AudioRecorder.rate,
					input=True,
					input_device_index=None if device_index < 0 else device_index,
					frames_per_buffer=AudioRecorder.chunk_size,
				)
				chunks = int(AudioRecorder.rate / AudioRecorder.chunk_size)  # ~1 second
				for _ in range(chunks):
					data = stream.read(AudioRecorder.chunk_size, exception_on_overflow=False)
					chunk_peak = calculate_peak_level(data)
					if chunk_peak > peak:
						peak = chunk_peak
				stream.stop_stream()
				stream.close()
			finally:
				pa.terminate()
			wx.CallAfter(self._show_sample_result, peak, None)
		except Exception as exc:
			wx.CallAfter(self._show_sample_result, 0, str(exc))

	def _show_sample_result(self, peak: int, error: str | None) -> None:
		self.sample_mic_button.SetLabel(_("Sa&mple microphone level"))
		self.sample_mic_button.Enable()
		if error:
			wx.MessageBox(
				_("Could not sample microphone: {}").format(error),
				_("Microphone sample"),
				wx.OK | wx.ICON_ERROR,
				self,
			)
			return
		suggestion = min(peak + 200, 32767)
		message = _(
			"Peak level during silence: {peak}\n\n"
			"Set your Silence threshold above this value so that "
			"quiet moments are detected correctly.\n\n"
			"Suggested threshold: {suggestion}"
		).format(peak=peak, suggestion=suggestion)
		dlg = wx.MessageDialog(
			self,
			message,
			_("Microphone sample"),
			wx.YES_NO | wx.ICON_INFORMATION,
		)
		dlg.SetYesNoLabels(_("Set threshold to {}").format(suggestion), _("Close"))
		if dlg.ShowModal() == wx.ID_YES:
			self.silence_threshold.SetValue(suggestion)
		dlg.Destroy()

	def on_get_api_key(self, _event) -> None:
		message = _(
			"To get a Groq API key:\n\n"
			"1. Log in or create a Groq account.\n"
			"2. Choose Create key.\n"
			"3. Follow the prompts.\n"
			"4. Copy the new key.\n"
			"5. Return to this settings panel and paste the key into the Groq API key field.\n\n"
			"Press OK to open the Groq keys page in your browser."
		)
		wx.MessageBox(
			message,
			_("Get Groq API key"),
			wx.OK | wx.ICON_INFORMATION,
			self,
		)
		webbrowser.open("https://console.groq.com/keys")

	def onSave(self) -> None:
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
		}
		config_manager.update_base_profile(values)
