import pathlib
import sys
import types
import unittest

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
	))

from audio_recorder import calculate_peak_level, calculate_lead_in_silence


class CalculatePeakLevelTests(unittest.TestCase):
	def test_empty_frame_returns_zero(self):
		self.assertEqual(calculate_peak_level(b""), 0)

	def test_detects_loudest_sample(self):
		frame = (100).to_bytes(2, "little", signed=True) + (-1200).to_bytes(2, "little", signed=True)
		self.assertEqual(calculate_peak_level(frame), 1200)


class CalculateLeadInSilenceTests(unittest.TestCase):
	def test_250ms_at_16000hz(self):
		silence = calculate_lead_in_silence(16000)
		num_samples = len(silence) // 2
		self.assertEqual(num_samples, 4000)

	def test_250ms_at_48000hz(self):
		silence = calculate_lead_in_silence(48000)
		num_samples = len(silence) // 2
		self.assertEqual(num_samples, 12000)

	def test_all_zeros(self):
		silence = calculate_lead_in_silence(16000)
		self.assertEqual(silence, b"\x00" * len(silence))

	def test_custom_duration(self):
		silence = calculate_lead_in_silence(16000, duration_ms=500)
		num_samples = len(silence) // 2
		self.assertEqual(num_samples, 8000)

	def test_stereo_doubles_size(self):
		silence_mono = calculate_lead_in_silence(16000, channels=1)
		silence_stereo = calculate_lead_in_silence(16000, channels=2)
		self.assertEqual(len(silence_stereo), len(silence_mono) * 2)


if __name__ == "__main__":
	unittest.main()
