import array
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "globalPlugins" / "groqVoiceDictation"
if str(MODULE_DIR) not in sys.path:
	sys.path.insert(0, str(MODULE_DIR))

from audio_processor import (
	calculate_rms,
	estimate_noise_floor,
	find_first_voice_sample,
	find_last_voice_sample,
	frame_peak,
	frame_rms,
	has_voice,
	trim_silence,
)


def _silence_bytes(num_samples: int, sample_width: int = 2) -> bytes:
	"""Build a ``bytes`` blob of N int16 zeros."""
	return b"\x00" * (num_samples * sample_width)


def _tone_bytes(num_samples: int, amplitude: int = 8000) -> bytes:
	"""Build a ``bytes`` blob of N int16 samples with a constant amplitude."""
	samples = array.array("h", [amplitude] * num_samples)
	return samples.tobytes()


class FramePeakTests(unittest.TestCase):
	def test_empty_returns_zero(self):
		self.assertEqual(frame_peak(b""), 0)

	def test_detects_loudest_sample(self):
		frame = (100).to_bytes(2, "little", signed=True) + (-1200).to_bytes(2, "little", signed=True)
		self.assertEqual(frame_peak(frame), 1200)

	def test_uses_absolute_value(self):
		frame = (-32767).to_bytes(2, "little", signed=True)
		self.assertEqual(frame_peak(frame), 32767)


class FrameRmsTests(unittest.TestCase):
	def test_empty_returns_zero(self):
		self.assertEqual(frame_rms(b""), 0.0)

	def test_constant_amplitude_equals_amplitude(self):
		# RMS of a constant signal is the absolute value of that signal.
		frame = (1000).to_bytes(2, "little", signed=True) * 50
		self.assertAlmostEqual(frame_rms(frame), 1000.0, places=3)

	def test_silence_returns_zero(self):
		self.assertEqual(frame_rms(_silence_bytes(100)), 0.0)


class CalculateRmsTests(unittest.TestCase):
	def test_joins_iterable_of_frames(self):
		frames = [_tone_bytes(50, 1000), _tone_bytes(50, 1000)]
		self.assertAlmostEqual(calculate_rms(frames), 1000.0, places=3)

	def test_accepts_single_bytes(self):
		self.assertAlmostEqual(calculate_rms(_tone_bytes(100, 2000)), 2000.0, places=3)

	def test_empty_returns_zero(self):
		self.assertEqual(calculate_rms(b""), 0.0)


class FindFirstVoiceSampleTests(unittest.TestCase):
	def test_empty_returns_minus_one(self):
		self.assertEqual(find_first_voice_sample(b"", threshold=200), -1)

	def test_all_silence_returns_minus_one(self):
		self.assertEqual(find_first_voice_sample(_silence_bytes(1000), threshold=200), -1)

	def test_returns_offset_of_first_loud_sample(self):
		# 100 samples of silence, then 5 samples of 8000
		frame = _silence_bytes(100) + _tone_bytes(5, 8000)
		self.assertEqual(find_first_voice_sample(frame, threshold=200), 100)

	def test_uses_threshold(self):
		# Signal of 500 amplitude: not above threshold 1000
		frame = _tone_bytes(50, 500)
		self.assertEqual(find_first_voice_sample(frame, threshold=1000), -1)
		# Same signal, lower threshold: detected
		self.assertEqual(find_first_voice_sample(frame, threshold=100), 0)

	def test_works_with_iterable_of_frames(self):
		# Same as the offset test, but the input is a list of frames
		# rather than one joined bytes blob. The recorder keeps its
		# audio in this shape; the helper must accept it.
		frames = [_silence_bytes(60), _silence_bytes(40), _tone_bytes(5, 8000)]
		self.assertEqual(find_first_voice_sample(frames, threshold=200), 100)


class FindLastVoiceSampleTests(unittest.TestCase):
	def test_empty_returns_minus_one(self):
		self.assertEqual(find_last_voice_sample(b"", threshold=200), -1)

	def test_all_silence_returns_minus_one(self):
		self.assertEqual(find_last_voice_sample(_silence_bytes(1000), threshold=200), -1)

	def test_returns_offset_just_past_last_loud_sample(self):
		# 5 samples of tone, then 100 samples of silence. The exclusive
		# end is 5 (one past the last tone sample at index 4).
		frame = _tone_bytes(5, 8000) + _silence_bytes(100)
		self.assertEqual(find_last_voice_sample(frame, threshold=200), 5)

	def test_works_with_iterable_of_frames(self):
		frames = [_tone_bytes(5, 8000), _silence_bytes(100)]
		self.assertEqual(find_last_voice_sample(frames, threshold=200), 5)


class HasVoiceTests(unittest.TestCase):
	def test_silence_returns_false(self):
		self.assertFalse(has_voice(_silence_bytes(1000)))

	def test_quiet_signal_returns_false(self):
		# 100 amplitude is below the default 200 floor
		self.assertFalse(has_voice(_tone_bytes(50, 100)))

	def test_loud_signal_returns_true(self):
		self.assertTrue(has_voice(_tone_bytes(50, 1000)))

	def test_custom_threshold(self):
		frame = _tone_bytes(50, 50)
		self.assertFalse(has_voice(frame, threshold=200))
		self.assertTrue(has_voice(frame, threshold=10))


class TrimSilenceTests(unittest.TestCase):
	RATE = 16000

	def test_all_silence_returns_empty(self):
		frame = _silence_bytes(self.RATE)  # 1 second of silence
		result = trim_silence(frame, rate=self.RATE, threshold=200)
		self.assertEqual(result, b"")

	def test_empty_returns_empty(self):
		self.assertEqual(trim_silence(b"", rate=self.RATE, threshold=200), b"")

	def test_trims_leading_silence_with_pad(self):
		# 500ms silence + 100ms tone + 200ms silence. With 100ms leading
		# pad, the result keeps 100ms of the leading silence + 100ms tone
		# + 200ms trailing silence = 400ms = 6400 samples = 12800 bytes.
		frame = (
			_silence_bytes(self.RATE // 2)  # 500ms
			+ _tone_bytes(self.RATE // 10, 8000)  # 100ms
			+ _silence_bytes(self.RATE // 5)  # 200ms
		)
		result = trim_silence(frame, rate=self.RATE, threshold=200,
			leading_pad_ms=100, trailing_pad_ms=200)
		# Total samples in result: 100ms (1600) + 100ms (1600) + 200ms (3200) = 6400
		self.assertEqual(len(result) // 2, 6400)

	def test_trims_trailing_silence(self):
		# 200ms silence + 100ms tone + 1.5s silence. With 50ms leading
		# pad and 100ms trailing pad, the result is 200ms-50ms=150ms lead
		# silence kept (samples 0..2400) ... wait, that's not right.
		# Correct: 100ms tone starts at sample 3200. 50ms leading pad
		# gives start = 3200 - 800 = 2400. Last tone at 4799. 100ms
		# trailing pad gives end = 4800 + 1600 = 6400. So result is
		# samples 2400..6400 = 4000 samples = 8000 bytes.
		frame = (
			_silence_bytes(self.RATE // 5)  # 200ms
			+ _tone_bytes(self.RATE // 10, 8000)  # 100ms
			+ _silence_bytes(int(self.RATE * 1.5))  # 1.5s
		)
		result = trim_silence(frame, rate=self.RATE, threshold=200,
			leading_pad_ms=50, trailing_pad_ms=100)
		self.assertEqual(len(result) // 2, 4000)

	def test_zero_pad_disables_trim(self):
		# 200ms silence + 100ms tone + 300ms silence. With 0 pad the
		# result should be exactly the tone portion (1600 samples).
		frame = (
			_silence_bytes(self.RATE // 5)
			+ _tone_bytes(self.RATE // 10, 8000)
			+ _silence_bytes(int(self.RATE * 0.3))
		)
		result = trim_silence(frame, rate=self.RATE, threshold=200,
			leading_pad_ms=0, trailing_pad_ms=0)
		self.assertEqual(len(result) // 2, self.RATE // 10)

	def test_works_with_iterable_of_frames(self):
		# Same as the leading-silence test, but input is a list of
		# small frames (mimicking what the recorder holds).
		frames = [
			_silence_bytes(800),  # 50ms
			_silence_bytes(8000),  # 500ms
			_tone_bytes(1600, 8000),  # 100ms
			_silence_bytes(3200),  # 200ms
		]
		result = trim_silence(frames, rate=self.RATE, threshold=200,
			leading_pad_ms=100, trailing_pad_ms=200)
		self.assertEqual(len(result) // 2, 6400)

	def test_short_utterance_kept_intact(self):
		# 100ms silence + 50ms tone + 100ms silence. With 100ms pads
		# the leading pad would overshoot the start, so the result
		# starts at 0; same on the back end. The tone is fully kept.
		frame = (
			_silence_bytes(self.RATE // 10)
			+ _tone_bytes(self.RATE // 20, 8000)
			+ _silence_bytes(self.RATE // 10)
		)
		result = trim_silence(frame, rate=self.RATE, threshold=200,
			leading_pad_ms=100, trailing_pad_ms=100)
		# Total samples: 1600 (lead, capped) + 800 (tone) + 1600 (trail, capped) = 4000
		self.assertEqual(len(result) // 2, 4000)

	def test_invalid_rate_raises(self):
		with self.assertRaises(ValueError):
			trim_silence(_tone_bytes(10, 8000), rate=0, threshold=200)

	def test_only_silence_in_middle_kept(self):
		# Two speech bursts separated by a long silence. Each is kept
		# with its own pad. The middle silence is preserved (not
		# trimmed) because the trim is only at the leading and trailing
		# edges of the audio.
		frame = (
			_silence_bytes(1600)  # 100ms
			+ _tone_bytes(1600, 8000)  # 100ms
			+ _silence_bytes(16000)  # 1s silence in the middle
			+ _tone_bytes(1600, 8000)  # 100ms
			+ _silence_bytes(1600)  # 100ms
		)
		result = trim_silence(frame, rate=self.RATE, threshold=200,
			leading_pad_ms=100, trailing_pad_ms=100)
		# 100ms lead + 100ms tone + 1s middle + 100ms tone + 100ms tail
		# = 1400ms = 22400 samples
		self.assertEqual(len(result) // 2, 22400)


class EstimateNoiseFloorTests(unittest.TestCase):
	def test_empty_returns_zero(self):
		self.assertEqual(estimate_noise_floor(b""), 0.0)

	def test_silence_returns_zero(self):
		self.assertEqual(estimate_noise_floor(_silence_bytes(1000)), 0.0)

	def test_quiet_signal_in_head(self):
		# 500ms of 200-amplitude signal at the head. RMS of a constant
		# signal equals its amplitude.
		frame = _tone_bytes(8000, 200)
		# head_only=True (default): looks at first 16KB which is the
		# whole frame here (16KB == 8000 int16 samples).
		self.assertAlmostEqual(estimate_noise_floor(frame), 200.0, places=2)

	def test_head_only_ignores_loud_tail(self):
		# Quiet head (200) then very loud tail (8000). The head-only
		# estimate should still report 200, not be skewed by the tail.
		frame = _tone_bytes(8000, 200) + _tone_bytes(8000, 8000)
		# 16KB cap means we look at the first 8000 samples (the quiet
		# part). The loud tail is beyond the cap and ignored.
		self.assertAlmostEqual(estimate_noise_floor(frame, head_only=True), 200.0, places=2)

	def test_head_only_false_uses_whole_signal(self):
		# Quiet head (200) then very loud tail (8000). With
		# head_only=False the estimate is over the whole signal.
		frame = _tone_bytes(8000, 200) + _tone_bytes(8000, 8000)
		rms = estimate_noise_floor(frame, head_only=False)
		# RMS of a signal that is half 200 and half 8000 is roughly
		# sqrt((200^2 + 8000^2) / 2) = sqrt(64200200) ~= 5663.
		self.assertGreater(rms, 5000)
		self.assertLess(rms, 6000)


if __name__ == "__main__":
	unittest.main()
