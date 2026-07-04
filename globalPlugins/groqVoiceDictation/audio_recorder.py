import array
import os
import tempfile
import threading
import time
import wave

import pyaudio
from logHandler import log


class AudioRecorderError(RuntimeError):
	pass


_SPEECH_FLOOR = 200
_SAMPLE_RATES = (16000, 44100, 48000, 8000, 22050)
_LEAD_IN_SILENCE_MS = 250

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
		input_device_index: int = -1,
		silence_enabled: bool = True,
		silence_timeout: int = 2,
		silence_threshold: int = 500,
		fallback_device_index: int = -1,
	) -> None:
		self._on_silence = on_silence
		self._input_device_index = input_device_index
		self._fallback_device_index = fallback_device_index
		self._silence_enabled = silence_enabled
		self._silence_timeout = silence_timeout
		self._silence_threshold = silence_threshold
		self._pa = None
		self._stream = None
		self._frames: list[bytes] = []
		self._lock = threading.Lock()
		self._silence_duration = 0.0
		self._silence_notified = False
		self._recording = False
		self._speech_detected = False
		self._used_fallback = False

	def start(self) -> None:
		if self._recording:
			return
		self._frames = []
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
					self._frames.append(calculate_lead_in_silence(rate, self.sample_width, self.channels))
					self._stream.start_stream()
					self._pa = pa
					self.rate = rate
					self._recording = True
					self._input_device_index = device_index
					elapsed_ms = (time.monotonic() - start_time) * 1000
					if primary_error is not None:
						self._used_fallback = True
						log.info("Fell back to microphone device %s at %s Hz after primary mic failed.", device_index, rate)
					log.info("AudioRecorder started in %.0fms (device=%d, rate=%d)", elapsed_ms, device_index, rate)
					return
				except Exception as exc:
					self._cleanup_stream()
					if device_error is None:
						device_error = exc
			if primary_error is None:
				primary_error = device_error
		raise AudioRecorderError(f"Could not open microphone: {primary_error}") from primary_error

	def stop(self) -> str:
		if not self._recording:
			raise AudioRecorderError("Recorder is not running.")
		self._recording = False
		try:
			if self._stream is not None:
				self._stream.stop_stream()
				self._stream.close()
		finally:
			self._stream = None
			self._pa = None
		return self._write_temp_wave()

	@property
	def is_recording(self) -> bool:
		return self._recording

	@property
	def used_fallback(self) -> bool:
		return self._used_fallback

	def has_speech(self) -> bool:
		with self._lock:
			joined = b"".join(self._frames)
		return calculate_peak_level(joined) > _SPEECH_FLOOR

	def _callback(self, in_data, frame_count, _time_info, _status):
		with self._lock:
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
		temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
		temp.close()
		with wave.open(temp.name, "wb") as wav_file:
			wav_file.setnchannels(self.channels)
			wav_file.setsampwidth(self.sample_width)
			wav_file.setframerate(self.rate)
			wav_file.writeframes(frames)
		return temp.name

	def _cleanup_stream(self) -> None:
		try:
			if self._stream is not None:
				self._stream.close()
		finally:
			self._stream = None
		self._pa = None
		self._recording = False

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
