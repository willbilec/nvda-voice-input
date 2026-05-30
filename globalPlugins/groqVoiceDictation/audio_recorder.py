import array
import os
import tempfile
import threading
import wave

import pyaudio


class AudioRecorderError(RuntimeError):
	pass


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
	) -> None:
		self._on_silence = on_silence
		self._input_device_index = input_device_index
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

	def start(self) -> None:
		if self._recording:
			return
		self._frames = []
		self._silence_duration = 0.0
		self._silence_notified = False
		self._speech_detected = False
		try:
			self._pa = pyaudio.PyAudio()
			self._stream = self._pa.open(
				format=pyaudio.paInt16,
				channels=self.channels,
				rate=self.rate,
				input=True,
				input_device_index=None if self._input_device_index < 0 else self._input_device_index,
				frames_per_buffer=self.chunk_size,
				stream_callback=self._callback,
			)
			self._stream.start_stream()
			self._recording = True
		except Exception as exc:
			self._cleanup_stream()
			raise AudioRecorderError(f"Could not open microphone: {exc}") from exc

	def stop(self) -> str:
		if not self._recording:
			raise AudioRecorderError("Recorder is not running.")
		self._recording = False
		try:
			if self._stream is not None:
				self._stream.stop_stream()
				self._stream.close()
		finally:
			if self._pa is not None:
				self._pa.terminate()
		self._stream = None
		self._pa = None
		return self._write_temp_wave()

	@property
	def is_recording(self) -> bool:
		return self._recording

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
		try:
			if self._pa is not None:
				self._pa.terminate()
		finally:
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
		devices: list[tuple[int, str]] = []
		for index in range(pa.get_device_count()):
			info = pa.get_device_info_by_index(index)
			if int(info.get("maxInputChannels", 0)) > 0:
				name = str(info.get("name", f"Microphone {index}"))
				devices.append((index, name))
		return devices
	finally:
		pa.terminate()
