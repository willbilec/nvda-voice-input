import json
import pathlib
import sys
import unittest
import types
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "globalPlugins" / "groqVoiceDictation"
if str(MODULE_DIR) not in sys.path:
	sys.path.insert(0, str(MODULE_DIR))

if "config" not in sys.modules:
	sys.modules["config"] = types.SimpleNamespace(conf=None, AggregatedSection=dict)
if "logHandler" not in sys.modules:
	sys.modules["logHandler"] = types.SimpleNamespace(log=types.SimpleNamespace(error=lambda *args, **kwargs: None))

from groq_client import build_cleanup_messages, map_http_error, normalize_api_key
import config_manager


class GroqClientHelpersTests(unittest.TestCase):
	def test_cleanup_messages_include_transcript(self):
		messages = build_cleanup_messages("hello there", "light")
		self.assertEqual(messages[0]["role"], "system")
		self.assertIn("hello there", messages[1]["content"])

	def test_heavy_cleanup_uses_rewrite_prompt(self):
		messages = build_cleanup_messages("draft", "heavy")
		self.assertIn("rewrite", messages[0]["content"].lower())

	def test_normalize_api_key_removes_whitespace(self):
		self.assertEqual(
			normalize_api_key(" gsk_test \r\n123 "),
			"gsk_test123",
		)

	def test_http_403_maps_to_auth_error(self):
		error = map_http_error(403, '{"error":{"message":"Forbidden"}}')
		self.assertEqual(error.category, "auth")
		self.assertEqual(error.message, "Forbidden")


class ConfigManagerTests(unittest.TestCase):
	def test_update_base_profile_updates_live_and_base_profile(self):
		live_section = {}
		base_section = {}
		fake_conf = mock.MagicMock()
		fake_conf.spec = {}
		fake_conf.__getitem__.side_effect = lambda key: live_section if key == config_manager.SECTION else KeyError(key)
		fake_conf.profiles = [{config_manager.SECTION: base_section}]
		with mock.patch.object(config_manager.config, "conf", fake_conf):
			config_manager.update_base_profile({"apiKey": "abc", "cleanupMode": "light"})
		self.assertEqual(live_section["apiKey"], "abc")
		self.assertEqual(live_section["cleanupMode"], "light")
		self.assertEqual(base_section["apiKey"], "abc")
		self.assertEqual(base_section["cleanupMode"], "light")


if __name__ == "__main__":
	unittest.main()
