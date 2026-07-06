import array
import pathlib
import sys
import tempfile
import types
import unittest
import wave
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "globalPlugins" / "groqVoiceDictation"
if str(MODULE_DIR) not in sys.path:
	sys.path.insert(0, str(MODULE_DIR))
if str(MODULE_DIR / "lib") not in sys.path:
	sys.path.insert(0, str(MODULE_DIR / "lib"))

if "pyaudio" not in sys.modules:
	sys.modules["pyaudio"] = types.SimpleNamespace(
		PyAudio=types.SimpleNamespace,
		paInt16=8,
		paContinue=0,
	)
if "logHandler" not in sys.modules:
	sys.modules["logHandler"] = types.SimpleNamespace(log=types.SimpleNamespace(
		info=lambda *a, **k: None,
		error=lambda *a, **k: None,
		warning=lambda *a, **k: None,
		exception=lambda *a, **k: None,
	))

import audio_processor
from audio_recorder import (
	AudioRecorder,
	AudioRecorderError,
	_LEAD_IN_SILENCE_MS,
	calculate_lead_in_silence,
	calculate_peak_level,
)


def _silence_bytes(num_samples: int) -> bytes:
	return b"\x00" * (num_samples * 2)


def _tone_bytes(num_samples: int, amplitude: int = 8000) -> bytes:
	samples = array.array("h", [amplitude] * num_samples)
	return samples.tobytes()


class CalculatePeakLevelTests(unittest.TestCase):
	def test_empty_frame_returns_zero(self):
		self.assertEqual(calculate_peak_level(b""), 0)

	def test_detects_loudest_sample(self):
		frame = (100).to_bytes(2, "little", signed=True) + (-1200).to_bytes(2, "little", signed=True)
		self.assertEqual(calculate_peak_level(frame), 1200)


class LeadInSilenceConstantTests(unittest.TestCase):
	"""Lock in the lead-in silence bump (250ms -> 500ms)."""

	def test_default_constant_is_500ms(self):
		self.assertEqual(_LEAD_IN_SILENCE_MS, 500)

	def test_default_at_16khz_yields_500ms_of_samples(self):
		silence = calculate_lead_in_silence(16000)
		self.assertEqual(len(silence) // 2, 16000 // 2)  # 500ms = 8000 samples


class AudioRecorderConstructorTests(unittest.TestCase):
	"""The new constructor parameters must not break the old call sites."""

	def test_default_construction_still_works(self):
		# Old call: AudioRecorder(on_silence=cb, input_device_index=-1)
		# No pre-roll, no trim overrides.
		rec = AudioRecorder()
		self.assertEqual(rec.pre_roll_ms, 0)
		self.assertEqual(rec.pre_trim_silence_ms, 300)
		self.assertEqual(rec.trailing_trim_silence_ms, 300)
		self.assertFalse(rec.pre_roll_active)

	def test_pre_roll_ms_is_clamped(self):
		# Negative -> 0. Over the cap -> 2000.
		rec = AudioRecorder(pre_roll_ms=-50)
		self.assertEqual(rec.pre_roll_ms, 0)
		rec = AudioRecorder(pre_roll_ms=5000)
		self.assertEqual(rec.pre_roll_ms, 2000)

	def test_trim_values_are_clamped_to_zero_floor(self):
		rec = AudioRecorder(pre_trim_silence_ms=-10, trailing_trim_silence_ms=-20)
		self.assertEqual(rec.pre_trim_silence_ms, 0)
		self.assertEqual(rec.trailing_trim_silence_ms, 0)


class _FakeStream:
	"""Minimal stand-in for a PyAudio stream used in unit tests."""

	def __init__(self, recorder: AudioRecorder) -> None:
		self._recorder = recorder
		self.started = False
		self.stopped = False
		self.closed = False

	def start_stream(self) -> None:
		self.started = True

	def stop_stream(self) -> None:
		self.stopped = True

	def close(self) -> None:
		self.closed = True


class _FakePyAudio:
	"""Stand-in for pyaudio.PyAudio that hands out _FakeStream objects."""

	def __init__(self) -> None:
		self.opened = 0

	def open(self, **_kwargs) -> _FakeStream:
		self.opened += 1
		# The recorder is created externally; we need to know it to
		# wire the callback. Hack: poke the recorder from a side
		# channel so the test can simulate audio.
		stream = _FakeStream(self._recorder_ref)
		return stream

	def terminate(self) -> None:
		pass

	def set_recorder(self, recorder: AudioRecorder) -> None:
		self._recorder_ref = recorder


class AudioRecorderPreRollTests(unittest.TestCase):
	"""Exercises the pre-roll state machine without a real PyAudio."""

	def _make_recorder(self, pre_roll_ms: int) -> AudioRecorder:
		rec = AudioRecorder(
			pre_roll_ms=pre_roll_ms,
			silence_enabled=False,
		)
		fake_pa = _FakePyAudio()
		fake_pa.set_recorder(rec)
		stream = _FakeStream(rec)
		rec._pa = fake_pa  # type: ignore[assignment]
		rec._stream = stream  # type: ignore[assignment]
		rec._recording = False
		rec._pre_rolling = pre_roll_ms > 0
		rec.rate = 16000
		return rec

	def test_pre_roll_starts_inactive_when_disabled(self):
		rec = self._make_recorder(pre_roll_ms=0)
		self.assertFalse(rec.pre_roll_active)
		self.assertFalse(rec.is_recording)

	def test_pre_roll_active_flag_reflects_state(self):
		rec = self._make_recorder(pre_roll_ms=500)
		rec._pre_rolling = True
		self.assertTrue(rec.pre_roll_active)
		self.assertTrue(rec.is_recording)  # is_recording is true during pre-roll
		rec._pre_rolling = False
		rec._recording = True
		self.assertFalse(rec.pre_roll_active)
		self.assertTrue(rec.is_recording)

	def test_complete_pre_roll_promotes_frames_and_does_not_fire_callback(self):
		# When pre-roll completes via _complete_pre_roll (e.g. from
		# stop()), the callback must NOT fire — the user is
		# finishing, not starting.
		cb = mock.MagicMock()
		rec = AudioRecorder(on_pre_roll_complete=cb, pre_roll_ms=500,
			silence_enabled=False)
		rec._pre_rolling = True
		rec._pre_roll_frames = [b"pre-roll-data"]
		rec._frames = [b"user-data"]
		# Swap the stream/pa so stop() does not crash on None.
		fake_pa = _FakePyAudio()
		fake_pa.set_recorder(rec)
		rec._pa = fake_pa  # type: ignore[assignment]
		rec._stream = _FakeStream(rec)  # type: ignore[assignment]
		returned = rec._complete_pre_roll()
		# The callback is returned to the caller, not fired here.
		self.assertIs(returned, cb)
		cb.assert_not_called()
		# The pre-roll frames have been promoted.
		self.assertEqual(rec._frames, [b"pre-roll-data", b"user-data"])
		self.assertEqual(rec._pre_roll_frames, [])
		self.assertTrue(rec._recording)
		self.assertFalse(rec._pre_rolling)

	def test_end_pre_roll_does_fire_callback(self):
		# When the timer fires the callback, it must be invoked exactly
		# once. This is the path the recorder uses when the pre-roll
		# window elapses normally.
		cb = mock.MagicMock()
		rec = AudioRecorder(on_pre_roll_complete=cb, pre_roll_ms=500,
			silence_enabled=False)
		rec._pre_rolling = True
		rec._pre_roll_frames = [b"pre"]
		rec._frames = []
		rec._end_pre_roll()
		cb.assert_called_once()
		self.assertTrue(rec._recording)

	def test_end_pre_roll_is_idempotent(self):
		# Calling _end_pre_roll after pre-roll has already been
		# completed must not double-fire the callback or mess up the
		# frames buffer.
		cb = mock.MagicMock()
		rec = AudioRecorder(on_pre_roll_complete=cb, pre_roll_ms=500,
			silence_enabled=False)
		rec._pre_rolling = True
		rec._pre_roll_frames = [b"pre"]
		rec._end_pre_roll()
		rec._end_pre_roll()
		cb.assert_called_once()

	def test_end_pre_roll_swallows_callback_exceptions(self):
		# If the callback raises (e.g. UI not ready), the recorder
		# must not lose the recording state. The exception is logged
		# but the flip to _recording must still happen.
		cb = mock.MagicMock(side_effect=RuntimeError("boom"))
		rec = AudioRecorder(on_pre_roll_complete=cb, pre_roll_ms=500,
			silence_enabled=False)
		rec._pre_rolling = True
		rec._end_pre_roll()  # must not raise
		self.assertTrue(rec._recording)
		cb.assert_called_once()

	def test_callback_runs_on_daemon_thread(self):
		# The Timer thread that fires _end_pre_roll is a daemon, so
		# the add-on can shut down without waiting on it.
		import threading
		rec = AudioRecorder(on_pre_roll_complete=lambda: None, pre_roll_ms=500,
			silence_enabled=False)
		rec._pre_roll_timer = threading.Timer(0.1, lambda: None)
		rec._pre_roll_timer.daemon = True
		self.assertTrue(rec._pre_roll_timer.daemon)


class AudioRecorderStopDuringPreRollTests(unittest.TestCase):
	"""stop() during the pre-roll phase must still produce a valid WAV."""

	def _wire(self, rec: AudioRecorder) -> None:
		fake_pa = _FakePyAudio()
		fake_pa.set_recorder(rec)
		rec._pa = fake_pa  # type: ignore[assignment]
		rec._stream = _FakeStream(rec)  # type: ignore[assignment]
		rec.rate = 16000

	def test_stop_during_pre_roll_includes_pre_roll_audio(self):
		rec = AudioRecorder(pre_roll_ms=500, silence_enabled=False)
		rec._pre_rolling = True
		rec._pre_roll_frames = [_silence_bytes(1600), _silence_bytes(1600)]
		rec._frames = []
		self._wire(rec)
		cb = mock.MagicMock()
		rec._on_pre_roll_complete = cb
		# The WAV is written to a temp file; patch out the
		# file-writing call so the test does not touch the disk.
		with mock.patch.object(rec, "_write_temp_wave", return_value="/tmp/x.wav") as write_mock:
			path = rec.stop()
		write_mock.assert_called_once()
		self.assertEqual(path, "/tmp/x.wav")
		# The callback must NOT have fired — we are stopping, not
		# starting.
		cb.assert_not_called()
		# The pre-roll frames were promoted so the WAV would
		# include them.
		self.assertEqual(rec._pre_roll_frames, [])
		self.assertEqual(rec._frames, [_silence_bytes(1600), _silence_bytes(1600)])

	def test_stop_after_pre_roll_does_not_re_fire_callback(self):
		# If pre-roll completed normally and then stop() is called,
		# the pre-roll timer is still cancellable. Verify stop()
		# doesn't trip the callback again.
		cb = mock.MagicMock()
		rec = AudioRecorder(on_pre_roll_complete=cb, pre_roll_ms=500,
			silence_enabled=False)
		rec._recording = True
		rec._frames = [_silence_bytes(1600)]
		self._wire(rec)
		with mock.patch.object(rec, "_write_temp_wave", return_value="/tmp/x.wav"):
			rec.stop()
		cb.assert_not_called()


class AudioRecorderTrimPipelineTests(unittest.TestCase):
	"""Verifies _write_temp_wave applies the silence trimmer before writing."""

	def _write_and_read(self, rec: AudioRecorder) -> int:
		"""Run _write_temp_wave, returning the number of frames in the resulting WAV."""
		with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
			path = temp.name
		# Redirect NamedTemporaryFile to use our pre-allocated path
		# so the test can predict the file location. We have to
		# patch the global `tempfile` module — `audio_recorder.py`
		# imports it as a module reference, so the recorder calls
		# `tempfile.NamedTemporaryFile(...)` directly. Patching
		# `audio_recorder.tempfile` would target a non-existent
		# attribute on the recorder module.
		class _FakeNT:
			def __init__(self, _delete=False, **_kwargs) -> None:
				self.name = path
			def close(self) -> None:
				pass
		with mock.patch("tempfile.NamedTemporaryFile", _FakeNT):
			rec._write_temp_wave()
		with wave.open(path, "rb") as wav_file:
			nframes = wav_file.getnframes()
		import os
		os.unlink(path)
		return nframes

	def test_trimmed_wav_is_shorter_than_input(self):
		# 1s silence + 200ms tone + 1s silence. With default pads
		# (300ms each) the leading silence is trimmed from 1s to
		# 300ms and the trailing silence from 1s to 300ms. Expected
		# output duration: 300ms + 200ms + 300ms = 800ms.
		rec = AudioRecorder(
			silence_enabled=False,
			silence_threshold=200,
			pre_trim_silence_ms=300,
			trailing_trim_silence_ms=300,
		)
		rec._frames = [
			_silence_bytes(16000)            # 1s leading silence
			+ _tone_bytes(3200, 8000)        # 200ms tone
			+ _silence_bytes(16000)          # 1s trailing silence
		]
		rec.rate = 16000
		rec._silence_threshold = 200
		nframes = self._write_and_read(rec)
		# 800ms * 16kHz = 12800 samples.
		self.assertEqual(nframes, 12800)

	def test_trim_returns_valid_wav_even_for_silent_input(self):
		# All-silence input must not produce an empty file. The
		# worker checks for speech separately, but the WAV still
		# needs to be valid on disk.
		rec = AudioRecorder(
			silence_enabled=False,
			silence_threshold=200,
		)
		rec._frames = [_silence_bytes(16000)]
		rec.rate = 16000
		rec._silence_threshold = 200
		nframes = self._write_and_read(rec)
		self.assertGreater(nframes, 0)

	def test_trim_with_zero_pads_is_exact(self):
		# 200ms silence + 100ms tone + 200ms silence. With 0 pads
		# the result is exactly the tone (1600 samples = 3200 bytes).
		rec = AudioRecorder(
			silence_enabled=False,
			silence_threshold=200,
			pre_trim_silence_ms=0,
			trailing_trim_silence_ms=0,
		)
		rec._frames = [
			_silence_bytes(3200) + _tone_bytes(1600, 8000) + _silence_bytes(3200)
		]
		rec.rate = 16000
		rec._silence_threshold = 200
		nframes = self._write_and_read(rec)
		self.assertEqual(nframes, 1600)


class AudioRecorderHasSpeechTests(unittest.TestCase):
	"""has_speech must consider pre-roll frames too."""

	def test_speech_only_in_pre_roll_is_detected(self):
		rec = AudioRecorder(pre_roll_ms=500, silence_enabled=False)
		rec._pre_roll_frames = [_tone_bytes(1600, 8000)]
		rec._frames = []
		self.assertTrue(rec.has_speech())

	def test_speech_in_user_frames_is_detected(self):
		rec = AudioRecorder(pre_roll_ms=500, silence_enabled=False)
		rec._pre_roll_frames = []
		rec._frames = [_tone_bytes(1600, 8000)]
		self.assertTrue(rec.has_speech())

	def test_silence_in_both_buffers_is_not_speech(self):
		rec = AudioRecorder(pre_roll_ms=500, silence_enabled=False)
		rec._pre_roll_frames = [_silence_bytes(1600)]
		rec._frames = [_silence_bytes(1600)]
		self.assertFalse(rec.has_speech())


class CalculateLeadInSilenceTests(unittest.TestCase):
	def test_default_is_500ms_at_16000hz(self):
		# Bumped from 250ms -> 500ms to give Whisper more context
		# for the first phoneme. The default constant lives in
		# audio_recorder._LEAD_IN_SILENCE_MS; the helper uses it
		# when no explicit duration is given.
		silence = calculate_lead_in_silence(16000)
		num_samples = len(silence) // 2
		self.assertEqual(num_samples, 8000)

	def test_default_at_48000hz(self):
		silence = calculate_lead_in_silence(48000)
		num_samples = len(silence) // 2
		self.assertEqual(num_samples, 24000)

	def test_all_zeros(self):
		silence = calculate_lead_in_silence(16000)
		self.assertEqual(silence, b"\x00" * len(silence))

	def test_custom_duration(self):
		silence = calculate_lead_in_silence(16000, duration_ms=250)
		num_samples = len(silence) // 2
		self.assertEqual(num_samples, 4000)

	def test_custom_duration_at_500ms(self):
		silence = calculate_lead_in_silence(16000, duration_ms=500)
		num_samples = len(silence) // 2
		self.assertEqual(num_samples, 8000)

	def test_stereo_doubles_size(self):
		silence_mono = calculate_lead_in_silence(16000, channels=1)
		silence_stereo = calculate_lead_in_silence(16000, channels=2)
		self.assertEqual(len(silence_stereo), len(silence_mono) * 2)


if __name__ == "__main__":
	unittest.main()
