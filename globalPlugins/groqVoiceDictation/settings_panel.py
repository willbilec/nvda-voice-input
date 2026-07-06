from typing import Any
import json
import threading
import webbrowser

import addonHandler
import config
from gui import guiHelper, nvdaControls
from gui.settingsDialogs import SettingsPanel
from logHandler import log
import wx

from .audio_recorder import AudioRecorder, calculate_peak_level, list_input_devices, _SAMPLE_RATES
from . import config_manager

try:
	addonHandler.initTranslation()
except addonHandler.AddonError:
	log.warning("Unable to init translations in settings panel.")


_addon = addonHandler.getCodeAddon()
ADDON_SUMMARY = _addon.manifest["summary"]


def ensure_device_in_choices(choices: list, device_index: int, label: str | None = None) -> bool:
	"""Add a saved device to a microphone choice list if it is missing.

	``list_input_devices`` filters out devices on host APIs it does not
	trust (MME, "Microsoft Sound Mapper", generic "Input", etc.), and
	devices that fail to enumerate on a particular run. If the user's
	saved device index is filtered out, the dialog cannot show it as the
	current selection — MicrophoneDialog._index_for_device returns 0 for
	missing entries, which silently shows "System default" as selected.
	Clicking OK then overwrites the user's saved device index with -1.

	This helper appends the saved index as an explicit entry so the
	dialog can show it as the active selection. Returns True if an entry
	was added, False otherwise (so the caller can log or count).
	"""
	if device_index < 0:
		return False
	if any(idx == device_index for idx, _ in choices):
		return False
	if label is None:
		label = _("Saved microphone (device %d, not currently detected)") % device_index
	log.warning(
		"Saved microphone device %d is not in the enumerated device list; "
		"adding it to the dialog choices so the user's selection is preserved.",
		device_index,
	)
	choices.append((device_index, label))
	return True


class APIKeyDialog(wx.Dialog):
	def __init__(self, parent: wx.Window, groq_key: str = "", gemini_key: str = "") -> None:
		super().__init__(parent, title=_("Manage API Keys"), size=(520, 320))
		panel = wx.Panel(self)
		sizer = wx.BoxSizer(wx.VERTICAL)

		groq_label = wx.StaticText(panel, label=_("Groq API Key"))
		groq_label.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
		sizer.Add(groq_label, 0, wx.LEFT | wx.TOP | wx.RIGHT, 10)

		groq_row = wx.BoxSizer(wx.HORIZONTAL)
		self._groq_key = wx.TextCtrl(panel, value=groq_key, style=wx.TE_PASSWORD)
		groq_row.Add(self._groq_key, 1, wx.EXPAND | wx.RIGHT, 8)
		groq_btn = wx.Button(panel, label=_("Get Groq API key"))
		groq_btn.Bind(wx.EVT_BUTTON, lambda _e: self._on_get_groq_key())
		groq_row.Add(groq_btn, 0)
		sizer.Add(groq_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

		gemini_label = wx.StaticText(panel, label=_("Gemini API Key"))
		gemini_label.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
		sizer.Add(gemini_label, 0, wx.LEFT | wx.TOP | wx.RIGHT, 10)

		gemini_row = wx.BoxSizer(wx.HORIZONTAL)
		self._gemini_key = wx.TextCtrl(panel, value=gemini_key, style=wx.TE_PASSWORD)
		gemini_row.Add(self._gemini_key, 1, wx.EXPAND | wx.RIGHT, 8)
		gemini_btn = wx.Button(panel, label=_("Get Gemini API key"))
		gemini_btn.Bind(wx.EVT_BUTTON, lambda _e: self._on_get_gemini_key())
		gemini_row.Add(gemini_btn, 0)
		sizer.Add(gemini_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

		note = wx.StaticText(
			panel,
			label=_("The Gemini key enables Gemini models for text cleanup (not transcription)."),
		)
		note.Wrap(480)
		sizer.Add(note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

		ok_btn = wx.Button(panel, wx.ID_OK, label=_("OK"))
		sizer.Add(ok_btn, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.BOTTOM, 10)

		panel.SetSizer(sizer)
		self._groq_key.SetFocus()

	@property
	def groq_key(self) -> str:
		return self._groq_key.GetValue().strip()

	@property
	def gemini_key(self) -> str:
		return self._gemini_key.GetValue().strip()

	def _on_get_groq_key(self) -> None:
		wx.MessageBox(
			_("To get a Groq API key:\n\n1. Log in or create a Groq account.\n2. Choose Create key.\n3. Copy the new key.\n4. Return here and paste the key.\n\nPress OK to open the Groq keys page."),
			_("Get Groq API key"), wx.OK | wx.ICON_INFORMATION, self,
		)
		webbrowser.open("https://console.groq.com/keys")

	def _on_get_gemini_key(self) -> None:
		wx.MessageBox(
			_("To get a Gemini API key:\n\n1. Go to Google AI Studio.\n2. Sign in with a Google account.\n3. Click 'Get API key'.\n4. Copy the key and paste it here.\n\nPress OK to open Google AI Studio."),
			_("Get Gemini API key"), wx.OK | wx.ICON_INFORMATION, self,
		)
		webbrowser.open("https://aistudio.google.com/apikey")


class TranscriptionDialog(wx.Dialog):
	def __init__(self, parent: wx.Window,
			provider: str, model: str, language: str,
			prompt_slots: list[str], active_slot: int) -> None:
		super().__init__(parent, title=_("Transcription"), size=(550, 400))
		panel = wx.Panel(self)
		sizer = wx.BoxSizer(wx.VERTICAL)
		helper = guiHelper.BoxSizerHelper(panel, sizer=sizer)

		self.transcription_provider = helper.addLabeledControl(
			_("&Provider:"), wx.Choice,
			choices=config_manager.label_list(config_manager.TRANSCRIPTION_PROVIDERS),
		)
		self.transcription_provider.SetSelection(0)

		self.transcription_model = helper.addLabeledControl(
			_("&Model:"), wx.Choice,
			choices=config_manager.TRANSCRIPTION_MODELS,
		)
		if model in config_manager.TRANSCRIPTION_MODELS:
			self.transcription_model.SetSelection(config_manager.TRANSCRIPTION_MODELS.index(model))
		else:
			self.transcription_model.SetSelection(0)

		self.transcription_language = helper.addLabeledControl(
			_("&Language:"), wx.Choice,
			choices=config_manager.label_list(config_manager.LANGUAGE_CHOICES),
		)
		self.transcription_language.SetSelection(
			config_manager.index_for_value(config_manager.LANGUAGE_CHOICES, language)
		)

		self._prompt_slots = list(prompt_slots)
		self._prompt_slot_index = active_slot
		self.prompt_slot_selector = helper.addLabeledControl(
			_("Prompt &slot:"), wx.Choice,
			choices=[_("Slot {}").format(i) for i in range(1, config_manager.PROMPT_SLOT_COUNT + 1)],
		)
		self.prompt_slot_selector.SetSelection(active_slot)
		self.prompt_slot_text = helper.addLabeledControl(
			_("Prompt &text (context for Whisper):"), wx.TextCtrl,
			value=self._prompt_slots[active_slot],
			style=wx.TE_MULTILINE,
		)
		self.prompt_slot_text.SetMinSize(wx.Size(-1, 80))

		ok_btn = helper.addItem(wx.Button(panel, wx.ID_OK, label=_("OK")))
		sizer.AddStretchSpacer()

		self.prompt_slot_selector.Bind(wx.EVT_CHOICE, self._on_slot_change)

	@property
	def model(self) -> str:
		sel = self.transcription_model.GetSelection()
		if 0 <= sel < len(config_manager.TRANSCRIPTION_MODELS):
			return config_manager.TRANSCRIPTION_MODELS[sel]
		return "whisper-large-v3-turbo"

	@property
	def language(self) -> str:
		return config_manager.LANGUAGE_CHOICES[self.transcription_language.GetSelection()][0]

	@property
	def prompt_slots(self) -> list[str]:
		self._prompt_slots[self._prompt_slot_index] = self.prompt_slot_text.GetValue()
		return self._prompt_slots

	@property
	def active_prompt_slot(self) -> int:
		return self.prompt_slot_selector.GetSelection()

	def _on_slot_change(self, _event) -> None:
		self._prompt_slots[self._prompt_slot_index] = self.prompt_slot_text.GetValue()
		self._prompt_slot_index = self.prompt_slot_selector.GetSelection()
		self.prompt_slot_text.SetValue(self._prompt_slots[self._prompt_slot_index])


class MicrophoneDialog(wx.Dialog):
	def __init__(self, parent: wx.Window, mic_choices: list, mic_device: int,
			fallback_device: int, fallback_enabled: bool) -> None:
		super().__init__(parent, title=_("Microphone"), size=(450, 230))
		panel = wx.Panel(self)
		sizer = wx.BoxSizer(wx.VERTICAL)
		helper = guiHelper.BoxSizerHelper(panel, sizer=sizer)

		self._mic_choices = mic_choices
		self.microphone_device = helper.addLabeledControl(
			_("&Microphone:"), wx.Choice,
			choices=[label for _, label in self._mic_choices],
		)
		self.microphone_device.SetSelection(self._index_for_device(mic_device))

		self.fallback_microphone_device = helper.addLabeledControl(
			_("&Fallback microphone:"), wx.Choice,
			choices=[label for _, label in self._mic_choices],
		)
		self.fallback_microphone_device.SetSelection(self._index_for_device(fallback_device))

		self.fallback_enabled = helper.addItem(
			wx.CheckBox(panel, label=_("Enable fallback &microphone"))
		)
		self.fallback_enabled.SetValue(fallback_enabled)

		helper.addItem(wx.Button(panel, wx.ID_OK, label=_("OK")))

	@property
	def mic_device(self) -> int:
		return self._mic_choices[self.microphone_device.GetSelection()][0]

	@property
	def fallback_device(self) -> int:
		return self._mic_choices[self.fallback_microphone_device.GetSelection()][0]

	@property
	def fallback_enabled_val(self) -> bool:
		return self.fallback_enabled.GetValue()

	def _index_for_device(self, device_index: int) -> int:
		for i, (idx, _) in enumerate(self._mic_choices):
			if idx == device_index:
				return i
		return 0


class SilenceDialog(wx.Dialog):
	def __init__(self, parent: wx.Window, mic_choices: list,
			enabled: bool, timeout: int, threshold: int, preflight: int,
			selected_mic: int) -> None:
		super().__init__(parent, title=_("Silence Detection"), size=(420, 320))
		panel = wx.Panel(self)
		sizer = wx.BoxSizer(wx.VERTICAL)
		helper = guiHelper.BoxSizerHelper(panel, sizer=sizer)

		self._mic_choices = mic_choices
		self._selected_mic = selected_mic

		self.silence_detection = helper.addItem(
			wx.CheckBox(panel, label=_("Enable &silence detection"))
		)
		self.silence_detection.SetValue(enabled)

		self.silence_timeout = helper.addLabeledControl(
			_("Silence timeout (&seconds):"), nvdaControls.SelectOnFocusSpinCtrl,
			value=str(timeout), min=1, max=15,
		)

		self.silence_threshold = helper.addLabeledControl(
			_("Silence &threshold:"), nvdaControls.SelectOnFocusSpinCtrl,
			value=str(threshold), min=100, max=32767,
		)

		self.sample_mic_button = helper.addItem(
			wx.Button(panel, label=_("Sa&mple microphone level"))
		)
		self.sample_mic_button.Bind(wx.EVT_BUTTON, self._on_sample)

		self.fallback_preflight = helper.addLabeledControl(
			_("Fallback preflight &wait (ms):"), nvdaControls.SelectOnFocusSpinCtrl,
			value=str(preflight), min=300, max=3000,
		)

		helper.addItem(wx.Button(panel, wx.ID_OK, label=_("OK")))

	@property
	def enabled(self) -> bool:
		return self.silence_detection.GetValue()

	@property
	def timeout(self) -> int:
		return int(self.silence_timeout.GetValue())

	@property
	def threshold(self) -> int:
		return int(self.silence_threshold.GetValue())

	@property
	def preflight(self) -> int:
		return int(self.fallback_preflight.GetValue())

	def _on_sample(self, _event) -> None:
		self.sample_mic_button.Disable()
		self.sample_mic_button.SetLabel(_("Sampling..."))
		device_index = self._selected_mic
		threading.Thread(target=self._do_sample, args=(device_index,), daemon=True).start()

	def _do_sample(self, device_index: int) -> None:
		import pyaudio
		peak = 0
		last_error = None
		try:
			pa = pyaudio.PyAudio()
			try:
				for rate in _SAMPLE_RATES:
					try:
						stream = pa.open(
							format=pyaudio.paInt16,
							channels=AudioRecorder.channels,
							rate=rate,
							input=True,
							input_device_index=None if device_index < 0 else device_index,
							frames_per_buffer=AudioRecorder.chunk_size,
						)
					except Exception as exc:
						last_error = exc
						continue
					try:
						chunks = int(rate / AudioRecorder.chunk_size)
						peak = 0
						for _ in range(chunks):
							data = stream.read(AudioRecorder.chunk_size, exception_on_overflow=False)
							cp = calculate_peak_level(data)
							if cp > peak:
								peak = cp
					finally:
						stream.stop_stream()
						stream.close()
					break
				else:
					if last_error is not None:
						raise last_error
			finally:
				pa.terminate()
			wx.CallAfter(self._show_result, peak, None)
		except Exception as exc:
			wx.CallAfter(self._show_result, 0, str(exc))

	def _show_result(self, peak: int, error: str | None) -> None:
		self.sample_mic_button.SetLabel(_("Sa&mple microphone level"))
		self.sample_mic_button.Enable()
		if error:
			wx.MessageBox(_("Could not sample microphone: {}").format(error),
				_("Microphone sample"), wx.OK | wx.ICON_ERROR, self)
			return
		suggestion = min(peak + 200, 32767)
		msg = _("Peak level during silence: {peak}\n\nSet your Silence threshold above this value.\n\nSuggested threshold: {suggestion}").format(peak=peak, suggestion=suggestion)
		dlg = wx.MessageDialog(self, msg, _("Microphone sample"), wx.YES_NO | wx.ICON_INFORMATION)
		dlg.SetYesNoLabels(_("Set threshold to {}").format(suggestion), _("Close"))
		if dlg.ShowModal() == wx.ID_YES:
			self.silence_threshold.SetValue(suggestion)
		dlg.Destroy()


class AudioDialog(wx.Dialog):
	"""Settings for the audio-processing pipeline.

	The knobs exposed here are independent of the silence-detection
	settings (which live in ``SilenceDialog``) so the user can tune
	"how is the WAV trimmed before it goes to Whisper" separately
	from "when does the recorder auto-stop". Both pipelines share
	the same ``silenceThreshold`` because the trim uses the same
	voice/no-voice boundary as the silence detector.
	"""

	def __init__(self, parent: wx.Window, pre_roll_ms: int, pre_trim_ms: int,
			trailing_trim_ms: int, auto_retry: bool) -> None:
		super().__init__(parent, title=_("Audio Processing"), size=(480, 360))
		panel = wx.Panel(self)
		sizer = wx.BoxSizer(wx.VERTICAL)
		helper = guiHelper.BoxSizerHelper(panel, sizer=sizer)

		intro = wx.StaticText(panel, label=_(
			"Fine-tune how the recorded audio is prepared for the "
			"Whisper transcription API. Defaults are tuned for "
			"typical desktop microphones."
		))
		intro.Wrap(440)
		helper.addItem(intro)

		self.pre_roll_ms = helper.addLabeledControl(
			_("Pre-roll &warm-up (ms):"), nvdaControls.SelectOnFocusSpinCtrl,
			value=str(pre_roll_ms), min=0, max=2000,
		)
		helper.addItem(wx.StaticText(panel, label=_(
			"0 disables pre-roll. 300-500ms helps capture the first "
			"phoneme on slow microphones (AirPods, USB headsets) at "
			"the cost of a small delay before the 'Listening' tone."
		))).Wrap(440)

		self.pre_trim_ms = helper.addLabeledControl(
			_("&Leading silence to keep (ms):"), nvdaControls.SelectOnFocusSpinCtrl,
			value=str(pre_trim_ms), min=0, max=2000,
		)
		helper.addItem(wx.StaticText(panel, label=_(
			"Trims long silence before the first word, but keeps "
			"this much as a buffer so Whisper has acoustic context. "
			"0 disables the trim."
		))).Wrap(440)

		self.trailing_trim_ms = helper.addLabeledControl(
			_("&Trailing silence to keep (ms):"), nvdaControls.SelectOnFocusSpinCtrl,
			value=str(trailing_trim_ms), min=0, max=2000,
		)
		helper.addItem(wx.StaticText(panel, label=_(
			"Trims silence after the last word, but keeps this "
			"much as a buffer. 0 disables the trim."
		))).Wrap(440)

		self.auto_retry = helper.addItem(
			wx.CheckBox(panel, label=_("&Auto-retry when first transcription looks suspicious"))
		)
		self.auto_retry.SetValue(auto_retry)
		helper.addItem(wx.StaticText(panel, label=_(
			"Retries without the prompt when the first pass returns "
			"a short result starting with a common opener. Recovers "
			"from the 'prompt-induced start-skipping' failure mode. "
			"Off skips the second API call but loses that recovery."
		))).Wrap(440)

		helper.addItem(wx.Button(panel, wx.ID_OK, label=_("OK")))

	@property
	def pre_roll(self) -> int:
		return int(self.pre_roll_ms.GetValue())

	@property
	def pre_trim(self) -> int:
		return int(self.pre_trim_ms.GetValue())

	@property
	def trailing_trim(self) -> int:
		return int(self.trailing_trim_ms.GetValue())

	@property
	def auto_retry_enabled(self) -> bool:
		return self.auto_retry.GetValue()


class CleanupDialog(wx.Dialog):
	# Reasoning effort choices shown in the dialog. Mirrors the
	# `reasoning_effort` parameter on Groq's GPT-OSS chat models.
	# Other models (Llama, Gemini) ignore the parameter, so the
	# dropdown is hidden for them in _on_model_change.
	REASONING_EFFORT_CHOICES: list[tuple[str, str]] = [
		("low", "Low (fastest)"),
		("medium", "Medium (balanced)"),
		("high", "High (most thorough)"),
	]
	REASONING_EFFORT_VALUES = [v for v, _ in REASONING_EFFORT_CHOICES]
	REASONING_EFFORT_DISPLAY = [label for _, label in REASONING_EFFORT_CHOICES]
	# Model id prefixes that honor `reasoning_effort`. Add to this set
	# when a new reasoning-capable model is onboarded.
	REASONING_CAPABLE_PREFIXES: tuple[str, ...] = ("openai/gpt-oss",)

	def __init__(self, parent: wx.Window, mode: str, model: str,
			gemini_key: str, reasoning_effort: str = "low") -> None:
		super().__init__(parent, title=_("Cleanup"), size=(520, 320))
		panel = wx.Panel(self)
		sizer = wx.BoxSizer(wx.VERTICAL)
		helper = guiHelper.BoxSizerHelper(panel, sizer=sizer)

		self.cleanup_mode = helper.addLabeledControl(
			_("&Cleanup mode:"), wx.Choice,
			choices=config_manager.label_list(config_manager.CLEANUP_MODES),
		)
		self.cleanup_mode.SetSelection(
			config_manager.index_for_value(config_manager.CLEANUP_MODES, mode)
		)

		self._model_values, self._model_display = self._build_model_choices(gemini_key)
		self.cleanup_model = helper.addLabeledControl(
			_("C&leanup model:"), wx.Choice,
			choices=self._model_display,
		)
		self._set_model_selection(model)

		self._llama_warning = helper.addItem(
			wx.StaticText(panel, label=_("Note: Llama models are deprecated and may be removed from Groq in the near future."))
		)
		self._llama_warning.SetForegroundColour(wx.Colour(180, 80, 0))

		# Reasoning-effort dropdown. Only enabled for models that
		# honor the parameter (currently gpt-oss). Hidden otherwise
		# to keep the dialog uncluttered.
		self.reasoning_effort = helper.addLabeledControl(
			_("Reasoning &effort:"), wx.Choice,
			choices=self.REASONING_EFFORT_DISPLAY,
		)
		effort_index = 0
		if reasoning_effort in self.REASONING_EFFORT_VALUES:
			effort_index = self.REASONING_EFFORT_VALUES.index(reasoning_effort)
		self.reasoning_effort.SetSelection(effort_index)
		self._reasoning_effort_note = helper.addItem(wx.StaticText(panel, label=_(
			"Higher = more model thinking, slower cleanup. Low is the "
			"recommended default and is enough for the rule-bound "
			"cleanup prompt. Bump to medium or high only if low "
			"misses a case."
		)))
		self._reasoning_effort_note.Wrap(440)

		self.cleanup_model.Bind(wx.EVT_CHOICE, self._on_model_change)
		helper.addItem(wx.Button(panel, wx.ID_OK, label=_("OK")))
		# Set the initial visibility for the saved model selection.
		self._on_model_change(None)

	@staticmethod
	def _build_model_choices(gemini_key: str) -> tuple[list[str], list[str]]:
		values: list[str] = []
		display: list[str] = []
		if gemini_key:
			for m in config_manager.GEMINI_CLEANUP_MODELS:
				values.append(m)
				rate = config_manager.GEMINI_CLEANUP_RATE_LIMITS.get(m, "?")
				display.append(_("{model} ({rate} req/day free)").format(model=m, rate=rate))
		for m in config_manager.CLEANUP_MODELS:
			values.append(m)
			display.append(m)
		return values, display

	@property
	def mode(self) -> str:
		return config_manager.CLEANUP_MODES[self.cleanup_mode.GetSelection()][0]

	@property
	def model(self) -> str:
		sel = self.cleanup_model.GetSelection()
		if 0 <= sel < len(self._model_values):
			return self._model_values[sel]
		return ""

	@property
	def reasoning_effort_value(self) -> str:
		sel = self.reasoning_effort.GetSelection()
		if 0 <= sel < len(self.REASONING_EFFORT_VALUES):
			return self.REASONING_EFFORT_VALUES[sel]
		return "low"

	def _set_model_selection(self, model: str) -> None:
		try:
			idx = self._model_values.index(model)
		except ValueError:
			idx = 0
		if idx < self.cleanup_model.GetCount():
			self.cleanup_model.SetSelection(idx)
		elif self.cleanup_model.GetCount() > 0:
			self.cleanup_model.SetSelection(0)

	def _is_reasoning_capable(self, model: str) -> bool:
		return any(model.startswith(p) for p in self.REASONING_CAPABLE_PREFIXES)

	def _on_model_change(self, _event) -> None:
		sel = self.cleanup_model.GetSelection()
		if 0 <= sel < len(self._model_values):
			model = self._model_values[sel]
		else:
			model = ""
		self._llama_warning.Show(model in config_manager.LLAMA_MODELS)
		# Show the reasoning-effort controls only when the selected
		# model actually honors the parameter. Other models ignore
		# it, so showing the dropdown would just be a UI lie.
		capable = self._is_reasoning_capable(model)
		self.reasoning_effort.Show(capable)
		note = getattr(self, "_reasoning_effort_note", None)
		if note is not None:
			note.Show(capable)
		# CleanupDialog is a wx.Dialog — it has no sizer of its own
		# (the sizer lives on the inner panel). Calling self.GetSizer()
		# returns None and crashes. self.Layout() re-lays out the
		# dialog's children instead.
		self.Layout()


class FeedbackDialog(wx.Dialog):
	def __init__(self, parent: wx.Window, feedback_mode: str, readback_mode: str,
			confirm_timeout: int) -> None:
		super().__init__(parent, title=_("Feedback & Readback"), size=(400, 240))
		panel = wx.Panel(self)
		sizer = wx.BoxSizer(wx.VERTICAL)
		helper = guiHelper.BoxSizerHelper(panel, sizer=sizer)

		self.feedback_mode = helper.addLabeledControl(
			_("&Feedback mode:"), wx.Choice,
			choices=config_manager.label_list(config_manager.FEEDBACK_MODES),
		)
		self.feedback_mode.SetSelection(
			config_manager.index_for_value(config_manager.FEEDBACK_MODES, feedback_mode)
		)

		self.readback_mode = helper.addLabeledControl(
			_("&Read-back mode:"), wx.Choice,
			choices=config_manager.label_list(config_manager.READBACK_MODES),
		)
		self.readback_mode.SetSelection(
			config_manager.index_for_value(config_manager.READBACK_MODES, readback_mode)
		)

		self.confirm_timeout = helper.addLabeledControl(
			_("Confirm &timeout (seconds):"), nvdaControls.SelectOnFocusSpinCtrl,
			value=str(confirm_timeout), min=2, max=15,
		)
		self.confirm_timeout.Enable(readback_mode == "confirm")

		self.readback_mode.Bind(wx.EVT_CHOICE, self._on_readback_change)

		helper.addItem(wx.Button(panel, wx.ID_OK, label=_("OK")))

	@property
	def feedback(self) -> str:
		return config_manager.FEEDBACK_MODES[self.feedback_mode.GetSelection()][0]

	@property
	def readback(self) -> str:
		return config_manager.READBACK_MODES[self.readback_mode.GetSelection()][0]

	@property
	def timeout(self) -> int:
		return int(self.confirm_timeout.GetValue())

	def _on_readback_change(self, _event) -> None:
		mode = config_manager.READBACK_MODES[self.readback_mode.GetSelection()][0]
		self.confirm_timeout.Enable(mode == "confirm")


class DebugDialog(wx.Dialog):
	def __init__(self, parent: wx.Window, paste_fallback: bool, speak_raw: bool) -> None:
		super().__init__(parent, title=_("Debugging"), size=(400, 160))
		panel = wx.Panel(self)
		sizer = wx.BoxSizer(wx.VERTICAL)
		helper = guiHelper.BoxSizerHelper(panel, sizer=sizer)

		self.allow_paste = helper.addItem(
			wx.CheckBox(panel, label=_("Allow &paste fallback when typing fails"))
		)
		self.allow_paste.SetValue(paste_fallback)

		self.speak_raw = helper.addItem(
			wx.CheckBox(panel, label=_("Speak ra&w transcript (diagnostic)"))
		)
		self.speak_raw.SetValue(speak_raw)

		helper.addItem(wx.Button(panel, wx.ID_OK, label=_("OK")))

	@property
	def paste_fallback(self) -> bool:
		return self.allow_paste.GetValue()

	@property
	def raw_transcript(self) -> bool:
		return self.speak_raw.GetValue()


class GroqVoiceDictationSettingsPanel(SettingsPanel):
	title = ADDON_SUMMARY

	def makeSettings(self, settingsSizer: wx.Sizer) -> None:
		sizer_helper = guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
		conf = config_manager.get()

		self._groq_key = conf["apiKey"]
		self._gemini_key = conf["geminiApiKey"]
		self._transcription_model = conf["transcriptionModel"]
		self._transcription_language = conf.get("transcriptionLanguage", "en")
		self._prompt_slots = config_manager.load_prompt_slots(conf)
		self._active_prompt_slot = int(conf.get("activePromptSlot", 0))
		self._mic_device = int(conf["microphoneDevice"])
		self._fallback_mic_device = int(conf["fallbackMicrophoneDevice"])
		self._fallback_enabled = conf["fallbackEnabled"]
		self._fallback_preflight = int(conf["fallbackPreflightMs"])
		self._silence_enabled = conf["silenceDetection"]
		self._silence_timeout = conf["silenceTimeout"]
		self._silence_threshold = conf["silenceThreshold"]
		self._cleanup_mode = conf["cleanupMode"]
		self._cleanup_model = conf["cleanupModel"]
		self._cleanup_reasoning_effort = conf.get("cleanupReasoningEffort", "low")
		self._feedback_mode = conf["feedbackMode"]
		self._readback_mode = conf["readbackMode"]
		self._confirm_timeout = conf["confirmTimeout"]
		self._paste_fallback = conf["allowPasteFallback"]
		self._speak_raw = conf["speakRawTranscript"]

		self._microphone_choices = [(-1, _("System default microphone"))]
		try:
			self._microphone_choices.extend(list_input_devices())
		except Exception:
			log.exception("Could not list microphone devices")
		# The user's saved microphones may not appear in the enumerated list
		# if they are on a host API that list_input_devices filters out (MME,
		# "Microsoft Sound Mapper", generic "Input", etc.), or if a device
		# was unplugged between sessions. Without this guard, opening the
		# microphone dialog would silently show "System default" as selected
		# (because _index_for_device returns 0 for missing entries), and
		# clicking OK would overwrite the user's saved device index with -1.
		# Inject the saved values as explicit entries so the dialog can show
		# them as the active selection.
		ensure_device_in_choices(self._microphone_choices, self._mic_device)
		ensure_device_in_choices(self._microphone_choices, self._fallback_mic_device)

		# Audio-processing knobs live in their own config keys; the
		# dedicated AudioDialog edits them. Read the current values
		# once so the dialog opens with the right state.
		from . import config_manager as _cm
		audio_cfg = _cm.get_audio_processing(conf)
		self._pre_roll_ms = audio_cfg["preRollMs"]
		self._pre_trim_ms = audio_cfg["preTrimSilenceMs"]
		self._trailing_trim_ms = audio_cfg["trailingTrimSilenceMs"]
		self._auto_retry = audio_cfg["autoRetryEnabled"]

		sizer_helper.addItem(
			wx.StaticText(self, label=_("Click a category below to manage settings:"))
		)

		self._api_key_btn = sizer_helper.addItem(
			wx.Button(self, label=_("Manage API Keys..."))
		)
		self._transcription_btn = sizer_helper.addItem(
			wx.Button(self, label=_("Transcription..."))
		)
		self._microphone_btn = sizer_helper.addItem(
			wx.Button(self, label=_("Microphone..."))
		)
		self._silence_btn = sizer_helper.addItem(
			wx.Button(self, label=_("Silence Detection..."))
		)
		self._audio_btn = sizer_helper.addItem(
			wx.Button(self, label=_("Audio Processing..."))
		)
		self._cleanup_btn = sizer_helper.addItem(
			wx.Button(self, label=_("Cleanup..."))
		)
		self._feedback_btn = sizer_helper.addItem(
			wx.Button(self, label=_("Feedback && Readback..."))
		)
		self._debug_btn = sizer_helper.addItem(
			wx.Button(self, label=_("Debugging..."))
		)

		sizer_helper.addItem(
			wx.StaticText(self,
				label=_("Use NVDA's installed add-ons dialog for this add-on's documentation."))
		)

		self._api_key_btn.Bind(wx.EVT_BUTTON, self._on_api_keys)
		self._transcription_btn.Bind(wx.EVT_BUTTON, self._on_transcription)
		self._microphone_btn.Bind(wx.EVT_BUTTON, self._on_microphone)
		self._silence_btn.Bind(wx.EVT_BUTTON, self._on_silence)
		self._audio_btn.Bind(wx.EVT_BUTTON, self._on_audio)
		self._cleanup_btn.Bind(wx.EVT_BUTTON, self._on_cleanup)
		self._feedback_btn.Bind(wx.EVT_BUTTON, self._on_feedback)
		self._debug_btn.Bind(wx.EVT_BUTTON, self._on_debug)

	def postInit(self) -> None:
		self._api_key_btn.SetFocus()

	def _on_api_keys(self, _event) -> None:
		dlg = APIKeyDialog(self, groq_key=self._groq_key, gemini_key=self._gemini_key)
		if dlg.ShowModal() == wx.ID_OK:
			self._groq_key = dlg.groq_key
			self._gemini_key = dlg.gemini_key
		dlg.Destroy()

	def _on_transcription(self, _event) -> None:
		dlg = TranscriptionDialog(self,
			provider="groq",
			model=self._transcription_model,
			language=self._transcription_language,
			prompt_slots=self._prompt_slots,
			active_slot=self._active_prompt_slot,
		)
		if dlg.ShowModal() == wx.ID_OK:
			self._transcription_model = dlg.model
			self._transcription_language = dlg.language
			self._prompt_slots = dlg.prompt_slots
			self._active_prompt_slot = dlg.active_prompt_slot
		dlg.Destroy()

	def _on_microphone(self, _event) -> None:
		dlg = MicrophoneDialog(self,
			mic_choices=self._microphone_choices,
			mic_device=self._mic_device,
			fallback_device=self._fallback_mic_device,
			fallback_enabled=self._fallback_enabled,
		)
		if dlg.ShowModal() == wx.ID_OK:
			self._mic_device = dlg.mic_device
			self._fallback_mic_device = dlg.fallback_device
			self._fallback_enabled = dlg.fallback_enabled_val
		dlg.Destroy()

	def _on_silence(self, _event) -> None:
		dlg = SilenceDialog(self,
			mic_choices=self._microphone_choices,
			enabled=self._silence_enabled,
			timeout=self._silence_timeout,
			threshold=self._silence_threshold,
			preflight=self._fallback_preflight,
			selected_mic=self._mic_device,
		)
		if dlg.ShowModal() == wx.ID_OK:
			self._silence_enabled = dlg.enabled
			self._silence_timeout = dlg.timeout
			self._silence_threshold = dlg.threshold
			self._fallback_preflight = dlg.preflight
		dlg.Destroy()

	def _on_audio(self, _event) -> None:
		dlg = AudioDialog(self,
			pre_roll_ms=self._pre_roll_ms,
			pre_trim_ms=self._pre_trim_ms,
			trailing_trim_ms=self._trailing_trim_ms,
			auto_retry=self._auto_retry,
		)
		if dlg.ShowModal() == wx.ID_OK:
			self._pre_roll_ms = dlg.pre_roll
			self._pre_trim_ms = dlg.pre_trim
			self._trailing_trim_ms = dlg.trailing_trim
			self._auto_retry = dlg.auto_retry_enabled
		dlg.Destroy()

	def _on_cleanup(self, _event) -> None:
		dlg = CleanupDialog(self,
			mode=self._cleanup_mode,
			model=self._cleanup_model,
			gemini_key=self._gemini_key,
			reasoning_effort=self._cleanup_reasoning_effort,
		)
		if dlg.ShowModal() == wx.ID_OK:
			self._cleanup_mode = dlg.mode
			self._cleanup_model = dlg.model
			self._cleanup_reasoning_effort = dlg.reasoning_effort_value
		dlg.Destroy()

	def _on_feedback(self, _event) -> None:
		dlg = FeedbackDialog(self,
			feedback_mode=self._feedback_mode,
			readback_mode=self._readback_mode,
			confirm_timeout=self._confirm_timeout,
		)
		if dlg.ShowModal() == wx.ID_OK:
			self._feedback_mode = dlg.feedback
			self._readback_mode = dlg.readback
			self._confirm_timeout = dlg.timeout
		dlg.Destroy()

	def _on_debug(self, _event) -> None:
		dlg = DebugDialog(self,
			paste_fallback=self._paste_fallback,
			speak_raw=self._speak_raw,
		)
		if dlg.ShowModal() == wx.ID_OK:
			self._paste_fallback = dlg.paste_fallback
			self._speak_raw = dlg.raw_transcript
		dlg.Destroy()

	def onSave(self) -> None:
		values: dict[str, Any] = {
			"apiKey": self._groq_key,
			"geminiApiKey": self._gemini_key,
			"transcriptionProvider": "groq",
			"transcriptionModel": self._transcription_model,
			"transcriptionLanguage": self._transcription_language,
			"promptSlots": json.dumps(self._prompt_slots, ensure_ascii=False),
			"activePromptSlot": self._active_prompt_slot,
			"cleanupMode": self._cleanup_mode,
			"cleanupModel": self._cleanup_model,
			"cleanupReasoningEffort": self._cleanup_reasoning_effort,
			"microphoneDevice": self._mic_device,
			"fallbackMicrophoneDevice": self._fallback_mic_device,
			"fallbackEnabled": self._fallback_enabled,
			"fallbackPreflightMs": self._fallback_preflight,
			"silenceDetection": self._silence_enabled,
			"silenceTimeout": self._silence_timeout,
			"feedbackMode": self._feedback_mode,
			"allowPasteFallback": self._paste_fallback,
			"silenceThreshold": self._silence_threshold,
			"readbackMode": self._readback_mode,
			"confirmTimeout": self._confirm_timeout,
			"speakRawTranscript": self._speak_raw,
			"preRollMs": self._pre_roll_ms,
			"preTrimSilenceMs": self._pre_trim_ms,
			"trailingTrimSilenceMs": self._trailing_trim_ms,
			"autoRetryEnabled": self._auto_retry,
		}
		config_manager.update_base_profile(values)
