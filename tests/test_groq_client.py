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

from groq_client import build_cleanup_messages, map_http_error, normalize_api_key, strip_thinking_tags, is_hallucination
import config_manager
from config_manager import load_prompt_slots, get_active_prompt


class GroqClientHelpersTests(unittest.TestCase):
	def test_cleanup_messages_include_transcript(self):
		messages = build_cleanup_messages("hello there", "light")
		self.assertEqual(messages[0]["role"], "system")
		self.assertIn("hello there", messages[1]["content"])

	def test_heavy_cleanup_uses_rewrite_prompt(self):
		messages = build_cleanup_messages("draft", "heavy")
		self.assertIn("transform", messages[0]["content"].lower())
		self.assertIn("restructure", messages[0]["content"].lower())

	def test_normalize_api_key_removes_whitespace(self):
		self.assertEqual(
			normalize_api_key(" gsk_test \r\n123 "),
			"gsk_test123",
		)

	def test_http_403_maps_to_auth_error(self):
		error = map_http_error(403, '{"error":{"message":"Forbidden"}}')
		self.assertEqual(error.category, "auth")
		self.assertEqual(error.message, "Forbidden")


class StripThinkingTagsTests(unittest.TestCase):
	def test_strips_think_tags(self):
		result = strip_thinking_tags(
			"<think>internal reasoning</think> cleaned text"
		)
		self.assertEqual(result, "cleaned text")

	def test_strips_thinking_tags(self):
		result = strip_thinking_tags(
			"<thinking>some reasoning</thinking> final output"
		)
		self.assertEqual(result, "final output")

	def test_strips_thought_tags(self):
		result = strip_thinking_tags(
			"<thought>reflection</thought> result"
		)
		self.assertEqual(result, "result")

	def test_strips_multiline_thinking(self):
		result = strip_thinking_tags(
			"<think>\nreasoning line 1\nreasoning line 2\n</think>\n\nthe actual text"
		)
		self.assertEqual(result, "the actual text")

	def test_no_thinking_tags_passes_through(self):
		result = strip_thinking_tags("plain text without any tags")
		self.assertEqual(result, "plain text without any tags")

	def test_empty_after_stripping_returns_empty(self):
		result = strip_thinking_tags("<think>only thinking</think>")
		self.assertEqual(result, "")

	def test_whitespace_after_stripping_is_preserved(self):
		result = strip_thinking_tags(
			"<think>x</think>  content  "
		)
		self.assertEqual(result, "content")


class IsHallucinationTests(unittest.TestCase):
	def test_thank_you_is_hallucination(self):
		self.assertTrue(is_hallucination("thank you"))
		self.assertTrue(is_hallucination("Thank You"))
		self.assertTrue(is_hallucination("  thank you  "))

	def test_thanks_for_watching_is_hallucination(self):
		self.assertTrue(is_hallucination("thanks for watching"))

	def test_normal_text_is_not_hallucination(self):
		self.assertFalse(is_hallucination("this issue needs to be fixed"))
		self.assertFalse(is_hallucination("hello world"))
		self.assertFalse(is_hallucination("please help with this bug"))

	def test_empty_is_hallucination(self):
		self.assertTrue(is_hallucination(""))
		self.assertTrue(is_hallucination("  "))

	def test_period_is_hallucination(self):
		self.assertTrue(is_hallucination("."))


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


class CleanupPromptRulesTests(unittest.TestCase):
	"""Lock in the anti-cutoff and anti-hallucination rules added to the prompts."""

	def test_all_modes_preserve_opening_words(self):
		for mode in ("light", "moderate", "heavy"):
			messages = build_cleanup_messages("anything goes", mode)
			content = messages[0]["content"]
			self.assertTrue(
				"PRESERVE THE FIRST WORD" in content or "PRESERVE OPENING WORDS" in content,
				msg=f"mode {mode!r} missing opening-word preservation rule",
			)

	def test_all_modes_clarify_false_start(self):
		for mode in ("light", "moderate", "heavy"):
			messages = build_cleanup_messages("anything goes", mode)
			self.assertIn(
				"NOT a false start",
				messages[0]["content"],
				msg=f"mode {mode!r} missing false-start clarification",
			)

	def test_light_and_moderate_forbid_adding_words(self):
		for mode in ("light", "moderate"):
			messages = build_cleanup_messages("anything goes", mode)
			content = messages[0]["content"]
			self.assertIn("Do NOT add any word", content)

	def test_light_forbids_inventing_preceding_word(self):
		messages = build_cleanup_messages("anything goes", "light")
		self.assertIn("Do NOT invent a preceding word", messages[0]["content"])

	def test_heavy_keeps_rewrite_keywords(self):
		messages = build_cleanup_messages("draft", "heavy")
		content = messages[0]["content"].lower()
		self.assertIn("transform", content)
		self.assertIn("restructure", content)
		self.assertIn("prefer words the speaker actually used", messages[0]["content"])

	def test_moderate_preserves_opening_words_every_sentence(self):
		messages = build_cleanup_messages("anything goes", "moderate")
		content = messages[0]["content"]
		self.assertIn("PRESERVE OPENING WORDS", content)
		self.assertIn("every sentence", content.lower())

	def test_moderate_forbids_pronoun_changes(self):
		messages = build_cleanup_messages("anything goes", "moderate")
		content = messages[0]["content"]
		self.assertIn("Do NOT change pronouns", content)

	def test_moderate_forbids_replacement(self):
		messages = build_cleanup_messages("anything goes", "moderate")
		self.assertIn("Do NOT replace any word with a different word", messages[0]["content"])

	def test_moderate_forbids_rephrasing(self):
		messages = build_cleanup_messages("anything goes", "moderate")
		self.assertIn("Do NOT rephrase", messages[0]["content"])

	def test_moderate_openers_include_yes_and_for_example(self):
		messages = build_cleanup_messages("anything goes", "moderate")
		content = messages[0]["content"]
		self.assertIn("'Yes'", content)
		self.assertIn("'For example'", content)

	def test_moderate_does_not_license_pronoun_clarity(self):
		messages = build_cleanup_messages("anything goes", "moderate")
		self.assertNotIn("pronoun clarity", messages[0]["content"].lower())

	def test_moderate_does_not_license_smoothing(self):
		messages = build_cleanup_messages("anything goes", "moderate")
		self.assertNotIn("smooth awkward", messages[0]["content"].lower())

	def test_all_modes_preserve_hedges(self):
		for mode in ("light", "moderate", "heavy"):
			messages = build_cleanup_messages("anything goes", mode)
			content = messages[0]["content"].lower()
			self.assertIn("hedges", content, msg=f"mode {mode!r} missing hedge preservation")
			self.assertIn("maybe", content, msg=f"mode {mode!r} hedge list missing 'maybe'")

	def test_all_modes_preserve_slang(self):
		for mode in ("light", "moderate", "heavy"):
			messages = build_cleanup_messages("anything goes", mode)
			content = messages[0]["content"].lower()
			self.assertIn("slang", content, msg=f"mode {mode!r} missing slang preservation")
			self.assertIn("fubar", content, msg=f"mode {mode!r} missing fubar example")

	def test_all_modes_forbid_answering_questions(self):
		for mode in ("light", "moderate", "heavy"):
			messages = build_cleanup_messages("anything goes", mode)
			content = messages[0]["content"].lower()
			self.assertIn("do not answer questions", content, msg=f"mode {mode!r} missing no-answer rule")
			self.assertIn("editor", content, msg=f"mode {mode!r} missing editor role framing")

	def test_all_modes_have_certainty_rule(self):
		for mode in ("light", "moderate", "heavy"):
			messages = build_cleanup_messages("anything goes", mode)
			content = messages[0]["content"]
			self.assertIn("CERTAINTY", content, msg=f"mode {mode!r} missing certainty rule")
			self.assertIn("RESPONSIBILITY", content)
			self.assertIn("EMOTION", content)

	def test_all_modes_protect_short_utterances(self):
		for mode in ("light", "moderate", "heavy"):
			messages = build_cleanup_messages("anything goes", mode)
			content = messages[0]["content"].lower()
			self.assertIn("short", content, msg=f"mode {mode!r} missing short-utterance rule")
			self.assertIn("under 8 words", content, msg=f"mode {mode!r} missing under-8-words threshold")

	def test_moderate_warns_about_paraphrase_creep(self):
		messages = build_cleanup_messages("anything goes", "moderate")
		content = messages[0]["content"]
		self.assertIn("PARAPHRASE CREEP", content)

	def test_heavy_fixes_asr_mishearings(self):
		messages = build_cleanup_messages("anything goes", "heavy")
		content = messages[0]["content"]
		self.assertIn("ASR mishearings", content)
		self.assertIn("TensorFlow", content)

	def test_moderate_does_not_fix_asr_mishearings(self):
		messages = build_cleanup_messages("anything goes", "moderate")
		self.assertNotIn("ASR mishearings", messages[0]["content"])

	def test_light_does_not_fix_asr_mishearings(self):
		messages = build_cleanup_messages("anything goes", "light")
		self.assertNotIn("ASR mishearings", messages[0]["content"])

	def test_heavy_forbids_pronoun_swaps(self):
		messages = build_cleanup_messages("anything goes", "heavy")
		self.assertIn("Do NOT change pronouns", messages[0]["content"])

	def test_speak_raw_transcript_in_confspec(self):
		self.assertIn("speakRawTranscript", config_manager.CONFSPEC)
		self.assertIn("boolean", config_manager.CONFSPEC["speakRawTranscript"])


class PromptSlotTests(unittest.TestCase):
	def test_load_prompt_slots_returns_defaults(self):
		slots = load_prompt_slots({})
		self.assertEqual(len(slots), config_manager.PROMPT_SLOT_COUNT)
		self.assertIn("NVDA", slots[0])
		self.assertIn("dictation", slots[1])
		self.assertIn("Python", slots[2])
		self.assertEqual(slots[9], "")

	def test_load_prompt_slots_from_json_string(self):
		custom = ["custom one", "custom two"]
		raw = json.dumps(custom)
		conf = {"promptSlots": raw}
		slots = load_prompt_slots(conf)
		self.assertEqual(slots[0], "custom one")
		self.assertEqual(slots[1], "custom two")
		self.assertEqual(len(slots), config_manager.PROMPT_SLOT_COUNT)

	def test_load_prompt_slots_from_list(self):
		custom = ["slot a", "slot b"]
		conf = {"promptSlots": custom}
		slots = load_prompt_slots(conf)
		self.assertEqual(slots[0], "slot a")
		self.assertEqual(slots[1], "slot b")

	def test_load_prompt_slots_truncates_excess(self):
		too_many = [f"slot {i}" for i in range(20)]
		conf = {"promptSlots": json.dumps(too_many)}
		slots = load_prompt_slots(conf)
		self.assertEqual(len(slots), config_manager.PROMPT_SLOT_COUNT)

	def test_load_prompt_slots_handles_invalid_json(self):
		conf = {"promptSlots": "not valid json[[["}
		slots = load_prompt_slots(conf)
		self.assertEqual(len(slots), config_manager.PROMPT_SLOT_COUNT)
		self.assertIn("NVDA", slots[0])

	def test_get_active_prompt_returns_slot_zero(self):
		slots = ["first", "second"]
		conf = {"promptSlots": json.dumps(slots), "activePromptSlot": 0}
		self.assertEqual(get_active_prompt(conf), "first")

	def test_get_active_prompt_returns_slot_one(self):
		slots = ["first", "second"]
		conf = {"promptSlots": json.dumps(slots), "activePromptSlot": 1}
		self.assertEqual(get_active_prompt(conf), "second")

	def test_get_active_prompt_out_of_range_returns_empty(self):
		slots = ["first", "second"]
		conf = {"promptSlots": json.dumps(slots), "activePromptSlot": 99}
		self.assertEqual(get_active_prompt(conf), "")

	def test_prompt_slots_in_confspec(self):
		self.assertIn("promptSlots", config_manager.CONFSPEC)
		self.assertIn("activePromptSlot", config_manager.CONFSPEC)


class CleanupDialogRegressionTests(unittest.TestCase):
	"""Regression tests for the settings-panel bug that caused the cleanup
	model choice not to stick. CleanupDialog is a wx.Dialog (not a
	SettingsPanel), so self.GetSizer() returns None and crashed with
	AttributeError on every model change, leaving the dialog in a state
	where OK clicks did not always save the new value. Fix: call
	self.Layout() (which re-lays out the dialog's children) instead.
	"""

	@classmethod
	def setUpClass(cls) -> None:
		cls._settings_panel_path = (
			ROOT / "globalPlugins" / "groqVoiceDictation" / "settings_panel.py"
		)
		cls._source = cls._settings_panel_path.read_text(encoding="utf-8")

	def test_on_model_change_does_not_call_getsizer(self):
		# The bug: CleanupDialog has no sizer of its own, so
		# self.GetSizer() returns None and .Layout() crashes with
		# AttributeError. The handler must not use GetSizer().Layout().
		self.assertNotIn(
			"self.GetSizer().Layout()",
			self._source,
			msg="CleanupDialog._on_model_change must not call self.GetSizer().Layout() "
			"because wx.Dialog has no sizer of its own — this crashes with "
			"AttributeError and prevents the model choice from being saved.",
		)

	def test_on_model_change_uses_layout(self):
		# The fix: call self.Layout() (re-lays out dialog children) instead.
		self.assertIn(
			"self.Layout()",
			self._source,
			msg="CleanupDialog._on_model_change should call self.Layout() to re-lay out "
			"the dialog after the llama-warning visibility changes.",
		)

	def test_on_model_change_mentions_cleanup_dialog_in_comment(self):
		# The comment near the fix should explain *why* the change is needed,
		# so the next maintainer doesn't revert it back to GetSizer().Layout().
		self.assertIn(
			"CleanupDialog is a wx.Dialog",
			self._source,
			msg="Add a comment near self.Layout() explaining that CleanupDialog is a "
			"wx.Dialog and has no sizer of its own.",
		)


if __name__ == "__main__":
	unittest.main()
