import array
import os
import tempfile
import threading
import time
import wave

import pyaudio
from logHandler import log

try:
	from . import audio_processor
except ImportError:  # pragma: no cover - top-level test import path
	import audio_processor


class AudioRecorderError(RuntimeError):
	pass


_SPEECH_FLOOR = 200
_SAMPLE_RATES = (16000, 44100, 48000, 8000, 22050)
# Lead-in silence (in ms) prepended to every recording. Whisper benefits
# from a small chunk of silence at the very start so the first phoneme
# is not lost on the 30-second sliding window boundary. 500ms matches
# the pre-pad used in the audio_processor.trim_silence call and the
# OpenAI cookbook's "milliseconds_until_sound" recipe.
_LEAD_IN_SILENCE_MS = 500
# Default values for the silence-trim pipeline. Pre-trim is the leading
# silence kept before the first voice sample; trailing-trim is the same
# on the back end. The defaults are deliberately generous so a brief
# pause before speaking is preserved — too small and the first word
# gets cut, too large and the recorded WAV carries seconds of nothing.
_DEFAULT_PRE_TRIM_MS = 300
_DEFAULT_TRAILING_TRIM_MS = 300
# Pre-roll cap. The recorder never sleeps for more than this on start()
# to capture warm-up audio. Exposed as a config upper bound in
# config_manager.
_MAX_PRE_ROLL_MS = 2000

_pa_instance = None
_pa_lock = threading.Lock()


def _get_pyaudio():
	global _pa_instance
	with _pa_lock:
		if _pa_instance is None:
			_pa_instance = pyaudio.PyAudio()
		return _pa_instance


def calculate_lead_in_silence(
	rate: int, sample_width: int = 2, channels: int = 1, duration_ms: int = _LEAD_IN_SILENCE_MS
) -> bytes:
	num_samples = int(duration_ms / 1000.0 * rate)
	return b"\x00" * (num_samples * sample_width * channels)


class AudioRecorder:
	rate = 16000
	channels = 1
	chunk_size = 1024
	sample_width = 2

	def __init__(
		self,
		on_silence=None,
		on_pre_roll_complete=None,
		input_device_index: int = -1,
		silence_enabled: bool = True,
		silence_timeout: int = 2,
		silence_threshold: int = 500,
		fallback_device_index: int = -1,
		pre_roll_ms: int = 0,
		pre_trim_silence_ms: int = _DEFAULT_PRE_TRIM_MS,
		trailing_trim_silence_ms: int = _DEFAULT_TRAILING_TRIM_MS,
	) -> None:
		"""Create a new audio recorder.

		Parameters
		----------
		on_silence:
			Callback fired once the silence timeout elapses after the last
			voice sample. The recorder must still be running when this fires
			(``stop()`` is what the callback will call).
		on_pre_roll_complete:
			Callback fired when the optional pre-roll window ends and the
			recorder actually starts capturing the user-facing audio. The
			typical caller plays a "Listening" tone from here. ``None``
			when pre-roll is disabled.
		pre_roll_ms:
			Milliseconds of audio to capture BEFORE the recorder is
			considered "started" (i.e. before the "Listening" tone plays
			and before silence detection runs). This warm-up window
			protects the first phoneme from being cut by mic wake-up
			latency, the OS audio subsystem coming online, etc. The
			captured pre-roll frames are prepended to the main recording
			in ``_write_temp_wave``. Default 0 (disabled) to preserve
			existing behaviour.
		pre_trim_silence_ms:
			Milliseconds of leading silence to keep when trimming the WAV
			that is sent to Whisper. Set to 0 to disable trimming.
		trailing_trim_silence_ms:
			Milliseconds of trailing silence to keep when trimming the WAV
			that is sent to Whisper. Set to 0 to disable trimming.
		"""
		self._on_silence = on_silence
		self._on_pre_roll_complete = on_pre_roll_complete
		self._input_device_index = input_device_index
		self._fallback_device_index = fallback_device_index
		self._silence_enabled = silence_enabled
		self._silence_timeout = silence_timeout
		self._silence_threshold = silence_threshold
		self._pre_roll_ms = max(0, min(int(pre_roll_ms), _MAX_PRE_ROLL_MS))
		self._pre_trim_silence_ms = max(0, int(pre_trim_silence_ms))
		self._trailing_trim_silence_ms = max(0, int(trailing_trim_silence_ms))
		self._pa = None
		self._stream = None
		self._frames: list[bytes] = []
		self._pre_roll_frames: list[bytes] = []
		self._pre_rolling = False
		self._pre_roll_timer: threading.Timer | None = None
		self._lock = threading.Lock()
		self._silence_duration = 0.0
		self._silence_notified = False
		self._recording = False
		self._speech_detected = False
		self._used_fallback = False

	def start(self) -> None:
		if self._recording or self._pre_rolling:
			return
		self._frames = []
		self._pre_roll_frames = []
		self._pre_rolling = False
		self._pre_roll_timer = None
		self._silence_duration = 0.0
		self._silence_notified = False
		self._speech_detected = False
		primary_error = None
		devices_to_try = [self._input_device_index]
		fallback = self._fallback_device_index if self._fallback_device_index != self._input_device_index else -2
		if fallback != -2:
			devices_to_try.append(fallback)
		start_time = time.monotonic()
		pa = None
		for device_index in devices_to_try:
			device_error = None
			for rate in _SAMPLE_RATES:
				try:
					if pa is None:
						pa = _get_pyaudio()
					self._stream = pa.open(
						format=pyaudio.paInt16,
						channels=self.channels,
						rate=rate,
						input=True,
						input_device_index=None if device_index < 0 else device_index,
						frames_per_buffer=self.chunk_size,
						stream_callback=self._callback,
					)
					self._stream.start_stream()
					self._pa = pa
					self.rate = rate
					self._input_device_index = device_index
					elapsed_ms = (time.monotonic() - start_time) * 1000
					if primary_error is not None:
						self._used_fallback = True
						log.info("Fell back to microphone device %s at %s Hz after primary mic failed.", device_index, rate)
					log.info(
						"AudioRecorder started in %.0fms (device=%d, rate=%d, pre_roll=%dms)",
						elapsed_ms, device_index, rate, self._pre_roll_ms,
					)
					# Two paths from here:
					#   * Pre-roll on: stream is open, frames go to
					#     _pre_roll_frames until the pre-roll timer fires.
					#     The "Listening" feedback is delayed until the
					#     pre-roll callback fires.
					#   * Pre-roll off: behave exactly as before. Add the
					#     fixed lead-in silence and flip _recording on so
					#     silence detection immediately starts.
					if self._pre_roll_ms > 0:
						self._pre_rolling = True
						self._pre_roll_timer = threading.Timer(
							self._pre_roll_ms / 1000.0,
							self._end_pre_roll,
						)
						self._pre_roll_timer.daemon = True
						self._pre_roll_timer.start()
					else:
						with self._lock:
							self._frames.append(
								calculate_lead_in_silence(rate, self.sample_width, self.channels)
							)
							self._recording = True
					return
				except Exception as exc:
					self._cleanup_stream()
					if device_error is None:
						device_error = exc
			if primary_error is None:
				primary_error = device_error
		raise AudioRecorderError(f"Could not open microphone: {primary_error}") from primary_error

	def _end_pre_roll(self) -> None:
		"""Promote the pre-roll frames into the main buffer and start recording.

		Runs on a daemon thread spawned by ``start()`` (when pre-roll is
		enabled). Atomically swaps the buffers under the recorder lock,
		flips ``_recording`` on, and fires the caller-provided
		``on_pre_roll_complete`` callback so the user hears the
		"Listening" feedback.
		"""
		callback = self._complete_pre_roll()
		if callback is not None:
			try:
				callback()
			except Exception:
				log.exception("on_pre_roll_complete callback raised")

	def _complete_pre_roll(self) -> object:
		"""Promote pre-roll frames under the lock; return the callback to fire.

		Separated from ``_end_pre_roll`` so ``stop()`` can promote the
		frames *without* firing the user-facing callback (the user has
		already heard what they need to hear and is now stopping the
		recording).
		"""
		with self._lock:
			if not self._pre_rolling:
				return None
			# Promote the pre-roll frames. The lead-in silence is skipped
			# because the pre-roll itself already covers the "before
			# speech" portion of the recording.
			self._frames = self._pre_roll_frames + self._frames
			self._pre_roll_frames = []
			self._pre_rolling = False
			self._recording = True
			callback = self._on_pre_roll_complete
		log.info("Pre-roll complete (%d frames promoted); recording started", len(self._frames))
		return callback

	@property
	def pre_roll_active(self) -> bool:
		return self._pre_rolling

	@property
	def pre_roll_ms(self) -> int:
		return self._pre_roll_ms

	@property
	def pre_trim_silence_ms(self) -> int:
		return self._pre_trim_silence_ms

	@property
	def trailing_trim_silence_ms(self) -> int:
		return self._trailing_trim_silence_ms

	@property
	def is_recording(self) -> bool:
		"""True when audio is being captured into the user-facing buffer.

		Returns True throughout both the pre-roll and the active recording
		phases. The pre-roll frames are still part of what will be
		returned by ``stop()``; treating pre-roll as "not recording" here
		would let the silence-detection callback fire mid-pre-roll and
		truncate the warm-up window.
		"""
		return self._recording or self._pre_rolling

	def stop(self) -> str:
		if not self._recording and not self._pre_rolling:
			raise AudioRecorderError("Recorder is not running.")
		# If we are still in the pre-roll phase, promote the pre-roll
		# frames now so the WAV that comes out of stop() includes them.
		# This handles the "user pressed the toggle and immediately
		# pressed it again" case — the audio that was captured during
		# the warm-up is still valuable and should be in the file.
		# Note: we use _complete_pre_roll (not _end_pre_roll) so the
		# on_pre_roll_complete callback is NOT fired from here. The
		# user is stopping the recording, not starting it; playing the
		# "Listening" tone at this point would be wrong.
		if self._pre_rolling:
			self._complete_pre_roll()
		self._recording = False
		# Cancel the pre-roll timer if it somehow survived (defensive —
		# _end_pre_roll should have cleared it via the lock-protected
		# branch, but if the timer fired after we entered stop() it
		# would still be live and try to touch the recorder).
		if self._pre_roll_timer is not None:
			try:
				self._pre_roll_timer.cancel()
			except Exception:
				pass
			self._pre_roll_timer = None
		# PortAudio's stop_stream() can hang and raise OSError [Errno -9987]
		# on some Windows audio drivers (we've seen this on Realtek and USB
		# devices that fail to drain the capture buffer). Without isolation
		# the exception propagates out of stop() and leaves the add-on in a
		# stuck "processing" state because _stop_and_process only catches
		# AudioRecorderError. Swallow OSError on each call so the close()
		# still runs and the caller always gets a wav path back.
		try:
			if self._stream is not None:
				try:
					self._stream.stop_stream()
				except OSError as exc:
					log.warning("PortAudio stop_stream failed: %s; forcing close", exc)
				try:
					self._stream.close()
				except OSError as exc:
					log.warning("PortAudio close failed: %s", exc)
		finally:
			self._stream = None
			self._pa = None
		return self._write_temp_wave()

	@property
	def used_fallback(self) -> bool:
		return self._used_fallback

	def has_speech(self) -> bool:
		with self._lock:
			# During pre-roll the user-facing frames buffer is empty;
			# we still want to know whether the pre-roll audio
			# contains any speech so the fallback-microphone logic
			# does not mis-fire.
			joined = b"".join(self._pre_roll_frames) + b"".join(self._frames)
		return calculate_peak_level(joined) > _SPEECH_FLOOR

	def _callback(self, in_data, frame_count, _time_info, _status):
		with self._lock:
			if self._pre_rolling:
				# Capture into the pre-roll buffer only. Silence detection
				# does not run during pre-roll: the user is not yet
				#"recording" from their perspective, and we do not want
				# the auto-stop callback to fire on the warm-up audio.
				self._pre_roll_frames.append(in_data)
			else:
				self._frames.append(in_data)
		if self._silence_enabled and self._recording:
			duration = frame_count / float(self.rate)
			peak = calculate_peak_level(in_data)
			if peak <= self._silence_threshold:
				self._silence_duration += duration
			else:
				self._speech_detected = True
				self._silence_duration = 0.0
				self._silence_notified = False
			if (
				self._speech_detected
				and not self._silence_notified
				and self._silence_duration >= self._silence_timeout
				and self._on_silence is not None
			):
				self._silence_notified = True
				threading.Thread(target=self._on_silence, daemon=True).start()
		return (None, pyaudio.paContinue)

	def _write_temp_wave(self) -> str:
		with self._lock:
			frames = b"".join(self._frames)
		# Apply silence trimming before the WAV hits disk. The trim
		# threshold is the same one the silence detector uses so the
		# "voice vs no-voice" boundary is consistent with the rest of
		# the pipeline. A trim that returns empty bytes means the
		# recorder never saw any audio above the threshold; in that
		# case the caller (``_process_recording``) already knows the
		# recording is silent via ``has_speech()`` and will not even
		# reach the transcription step.
		trimmed = audio_processor.trim_silence(
			frames,
			rate=self.rate,
			threshold=self._silence_threshold,
			leading_pad_ms=self._pre_trim_silence_ms,
			trailing_pad_ms=self._trailing_trim_silence_ms,
		)
		if not trimmed:
			# No speech at all. Still write an empty-but-valid WAV so
			# downstream code (which may inspect the file) does not see
			# a missing file. The worker treats an empty file the same
			# as no-speech and skips the API call.
			trimmed = b"\x00\x00" * self.rate  # 1s of silence
		pre_trim_seconds = len(frames) / float(self.rate * self.sample_width * self.channels)
		post_trim_seconds = len(trimmed) / float(self.rate * self.sample_width * self.channels)
		log.info(
			"AudioRecorder trim: %.2fs -> %.2fs (threshold=%d, pads=%d/%d ms)",
			pre_trim_seconds, post_trim_seconds, self._silence_threshold,
			self._pre_trim_silence_ms, self._trailing_trim_silence_ms,
		)
		temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
		temp.close()
		with wave.open(temp.name, "wb") as wav_file:
			wav_file.setnchannels(self.channels)
			wav_file.setsampwidth(self.sample_width)
			wav_file.setframerate(self.rate)
			wav_file.writeframes(trimmed)
		return temp.name

	def _cleanup_stream(self) -> None:
		if self._pre_roll_timer is not None:
			try:
				self._pre_roll_timer.cancel()
			except Exception:
				pass
			self._pre_roll_timer = None
		try:
			if self._stream is not None:
				self._stream.close()
		finally:
			self._stream = None
		self._pa = None
		self._recording = False
		self._pre_rolling = False

	@staticmethod
	def delete_file(path: str) -> None:
		if path and os.path.exists(path):
			os.remove(path)


def calculate_peak_level(frame_bytes: bytes) -> int:
	if not frame_bytes:
		return 0
	samples = array.array("h")
	samples.frombytes(frame_bytes)
	if not samples:
		return 0
	return max(abs(sample) for sample in samples)


def list_input_devices() -> list[tuple[int, str]]:
	pa = pyaudio.PyAudio()
	try:
		host_api_priority: dict[int, int] = {}
		for i in range(pa.get_host_api_count()):
			try:
				info = pa.get_host_api_info_by_index(i)
				host_api_priority[i] = _host_api_rank(str(info.get("name", "")))
			except Exception:
				host_api_priority[i] = 99
		_SKIP_PREFIXES = (
			"microsoft sound mapper",
			"primary sound capture driver",
		)
		raw: list[tuple[str, int, str]] = []
		for index in range(pa.get_device_count()):
			try:
				info = pa.get_device_info_by_index(index)
			except Exception:
				continue
			if int(info.get("maxInputChannels", 0)) <= 0:
				continue
			name = str(info.get("name", f"Microphone {index}")).strip()
			api_index = int(info.get("hostApi", -1))
			rank = host_api_priority.get(api_index, 99)
			if rank >= 99:
				continue
			if name.lower().startswith(_SKIP_PREFIXES):
				continue
			if name.lower() == "input" or name.lower().startswith("input (@"):
				continue
			base = _device_base_name(name)
			raw.append((base, rank, name, index))
		best: dict[str, tuple[int, str]] = {}
		for base, rank, name, index in raw:
			key = base.lower()
			if key not in best or rank < best[key][0] or (rank == best[key][0] and len(name) > len(best[key][1])):
				best[key] = (rank, name, index)
		result: list[tuple[int, str]] = []
		for key, (rank, name, index) in best.items():
			result.append((index, name))
		result.sort(key=lambda e: e[1].lower())
		return result
	finally:
		pa.terminate()


def _device_base_name(name: str) -> str:
	idx = name.find("(")
	if idx > 0:
		return name[:idx].strip()
	return name


def _host_api_rank(name: str) -> int:
	lower = name.lower()
	if "wasapi" in lower:
		return 0
	if "directsound" in lower:
		return 1
	if "mme" in lower:
		return 99
	return 99
