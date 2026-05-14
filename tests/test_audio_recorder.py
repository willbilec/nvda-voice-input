import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "globalPlugins" / "groqVoiceDictation"
if str(MODULE_DIR) not in sys.path:
	sys.path.insert(0, str(MODULE_DIR))
if str(MODULE_DIR / "lib") not in sys.path:
	sys.path.insert(0, str(MODULE_DIR / "lib"))

from audio_recorder import calculate_peak_level


class CalculatePeakLevelTests(unittest.TestCase):
	def test_empty_frame_returns_zero(self):
		self.assertEqual(calculate_peak_level(b""), 0)

	def test_detects_loudest_sample(self):
		frame = (100).to_bytes(2, "little", signed=True) + (-1200).to_bytes(2, "little", signed=True)
		self.assertEqual(calculate_peak_level(frame), 1200)


if __name__ == "__main__":
	unittest.main()
