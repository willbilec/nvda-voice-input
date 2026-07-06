import json
import pathlib
import sys
import tempfile
import unittest
import types
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "globalPlugins" / "groqVoiceDictation"
PACKAGE_PARENT = MODULE_DIR.parent
PLUGIN_PKG = "globalPlugins.groqVoiceDictation"
if str(MODULE_DIR) not in sys.path:
	sys.path.insert(0, str(MODULE_DIR))
# gemini_client uses a relative import (from .groq_client import
# strip_thinking_tags) so it must be loaded as part of the package,
# not as a top-level module.
if str(PACKAGE_PARENT.parent) not in sys.path:
	sys.path.insert(0, str(PACKAGE_PARENT.parent))

if "config" not in sys.modules:
	sys.modules["config"] = types.SimpleNamespace(conf=None, AggregatedSection=dict)
if "logHandler" not in sys.modules:
	sys.modules["logHandler"] = types.SimpleNamespace(
		log=types.SimpleNamespace(
			error=lambda *args, **kwargs: None,
			warning=lambda *args, **kwargs: None,
			info=lambda *args, **kwargs: None,
		)
	)

from groq_client import (
	GroqClient,
	GroqClientError,
	_looks_suspicious,
	_filter_hallucinated_segments,
	build_cleanup_messages,
	is_hallucination,
	map_http_error,
	normalize_api_key,
	strip_thinking_tags,
)
import importlib
# Make sure the plugin package is loaded so gemini_client's relative
# import resolves. Drop any cached copies so we get a clean import.
for _n in (PLUGIN_PKG, PACKAGE_PARENT.name):
	sys.modules.pop(_n, None)
sys.modules.pop("globalPlugins.groqVoiceDictation.gemini_client", None)
sys.modules.pop("globalPlugins.groqVoiceDictation.groq_client", None)
import globalPlugins  # noqa: F401
import globalPlugins.groqVoiceDictation  # noqa: F401
gemini_client = importlib.import_module(f"{PLUGIN_PKG}.gemini_client")
import config_manager
from config_manager import (
	get_active_prompt,
	get_audio_processing,
	load_prompt_slots,
)


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

	def test_moderate_forbids_pronoun_changes_for_clarity(self):
		"""Moderate does NOT change pronouns for "clarity" or "consistency"
		— the only allowed pronoun change is an ASR mishearing with
		unambiguous context.
		"""
		messages = build_cleanup_messages("anything goes", "moderate")
		content = messages[0]["content"]
		self.assertIn("Do NOT change a pronoun", content)
		self.assertIn("clarity", content.lower())

	def test_moderate_forbids_rephrasing(self):
		messages = build_cleanup_messages("anything goes", "moderate")
		content = messages[0]["content"]
		# The prompt forbids rephrasing; the exact phrasing is allowed
		# to vary as long as both the paraphrase and rephrase
		# prohibitions are present.
		self.assertIn("paraphrase", content.lower())
		self.assertIn("rephrase", content.lower())

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

	def test_light_does_not_fix_asr_mishearings(self):
		messages = build_cleanup_messages("anything goes", "light")
		self.assertNotIn("ASR mishearings", messages[0]["content"])

	def test_moderate_allows_asr_mishearing_fixes(self):
		"""Moderate has a narrow ASR-mishearing license: it may replace a
		word ONLY when the surrounding context makes the original
		clearly wrong. Light does not, and heavy has a broader license
		that includes restructuring.
		"""
		content = build_cleanup_messages("anything goes", "moderate")[0]["content"]
		self.assertIn("ASR mishearings", content)
		# Specific categories the prompt enumerates
		for phrase in (
			"Pronoun mishears",
			"Homophones that change meaning",
			"Compound terms and proper nouns",
		):
			self.assertIn(phrase, content, msg=f"moderate missing {phrase!r}")
		# Concrete examples the prompt should teach from
		for example in (
			"'tensor flow' -> 'TensorFlow'",
			"'API gate way' -> 'API gateway'",
			"'post gress SQL' -> 'PostgreSQL'",
		):
			self.assertIn(example, content, msg=f"moderate missing example {example!r}")
		# The "context makes the original clearly wrong" gate
		self.assertIn("context makes the original clearly wrong", content)
		# The "human transcriber" test that gates each fix
		self.assertIn("human transcriber", content)
		# Asymmetry rule: false positives worse than false negatives
		self.assertIn("false-positive fixes", content)

	def test_moderate_protects_short_utterances_from_asr_fixes(self):
		"""The ASR license is suspended for short utterances — too risky
		to correct when there's not enough context to confirm.
		"""
		content = build_cleanup_messages("anything goes", "moderate")[0]["content"]
		self.assertIn("under 8 words", content)
		# And the prompt must explicitly say NO ASR fix on short utterances
		self.assertIn("do NOT apply any ASR fix", content)

	def test_moderate_caps_asr_fixes_to_prevent_paraphrase_creep(self):
		"""The prompt must include a numerical cap on ASR fixes to stop
		the model from drifting into wholesale paraphrasing.
		"""
		content = build_cleanup_messages("anything goes", "moderate")[0]["content"]
		self.assertIn("5-10%", content)

	def test_heavy_forbids_pronoun_swaps(self):
		messages = build_cleanup_messages("anything goes", "heavy")
		self.assertIn("Do NOT change pronouns", messages[0]["content"])

	def test_speak_raw_transcript_in_confspec(self):
		self.assertIn("speakRawTranscript", config_manager.CONFSPEC)
		self.assertIn("boolean", config_manager.CONFSPEC["speakRawTranscript"])

	def test_cleanup_reasoning_effort_in_confspec(self):
		"""The reasoning-effort knob must be in the confspec so NVDA
		persists it across sessions. Default "low" gives new users
		the fast path; the Cleanup dialog can override it to
		medium/high for harder cases.
		"""
		self.assertIn("cleanupReasoningEffort", config_manager.CONFSPEC)
		spec = config_manager.CONFSPEC["cleanupReasoningEffort"]
		self.assertIn("string", spec)
		self.assertIn('"low"', spec,
			msg="Default must be 'low' so new users get the fast path")


class PromptSlotTests(unittest.TestCase):
	def test_load_prompt_slots_returns_defaults(self):
		slots = load_prompt_slots({})
		self.assertEqual(len(slots), config_manager.PROMPT_SLOT_COUNT)
		# The trimmed defaults use small glossaries to reduce the
		# prompt-induced start-skipping failure mode. Each slot
		# should contain a representative term from its category.
		self.assertIn("Python", slots[0])
		self.assertIn("dictation", slots[1])
		self.assertIn("PostgreSQL", slots[2])
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
		# Falls back to the defaults — slot 0 contains "Python".
		self.assertIn("Python", slots[0])

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


class CleanupPostBodyTests(unittest.TestCase):
	"""Pin the wire shape of the cleanup POST.

	The cleanup call is on the hot path: it runs on every dictation
	that isn't mode=raw. Getting the body right (max tokens cap,
	reasoning effort, no leaked thinking tokens) is the speedup.
	"""

	def _ok_response(self, content: str = "cleaned text") -> mock.MagicMock:
		resp = mock.MagicMock()
		resp.ok = True
		resp.json.return_value = {
			"choices": [{"message": {"content": content}}],
			"usage": {},
		}
		return resp

	def _client(self) -> GroqClient:
		return GroqClient(api_key="test_key")

	def test_gpt_oss_cleanup_uses_low_reasoning_effort_by_default(self):
		client = self._client()
		with mock.patch.object(client._session, "post",
				return_value=self._ok_response()) as post_mock:
			client.cleanup("raw transcript", "moderate", "openai/gpt-oss-20b")
		body = post_mock.call_args.kwargs["json"]
		self.assertEqual(body["reasoning_effort"], "low")

	def test_gpt_oss_cleanup_accepts_custom_reasoning_effort(self):
		client = self._client()
		with mock.patch.object(client._session, "post",
				return_value=self._ok_response()) as post_mock:
			client.cleanup("raw", "moderate", "openai/gpt-oss-20b",
				reasoning_effort="high")
		body = post_mock.call_args.kwargs["json"]
		self.assertEqual(body["reasoning_effort"], "high")

	def test_gpt_oss_120b_also_honors_reasoning_effort(self):
		client = self._client()
		with mock.patch.object(client._session, "post",
				return_value=self._ok_response()) as post_mock:
			client.cleanup("raw", "moderate", "openai/gpt-oss-120b",
				reasoning_effort="medium")
		body = post_mock.call_args.kwargs["json"]
		self.assertEqual(body["reasoning_effort"], "medium")

	def test_non_gpt_oss_models_skip_reasoning_effort(self):
		"""reasoning_effort is only honored by gpt-oss models. Sending
		it to Llama or Qwen would be a no-op at best and a warning
		at worst, so the client must omit it for other models.
		"""
		client = self._client()
		for model in (
			"llama-3.1-8b-instant",
			"llama-3.3-70b-versatile",
			"qwen/qwen3-32b",
			"meta-llama/llama-4-scout-17b-16e-instruct",
		):
			with mock.patch.object(client._session, "post",
					return_value=self._ok_response()) as post_mock:
				client.cleanup("raw", "moderate", model)
			body = post_mock.call_args.kwargs["json"]
			self.assertNotIn("reasoning_effort", body,
				msg=f"reasoning_effort must not be sent to {model}")

	def test_cleanup_caps_max_completion_tokens(self):
		"""No max_tokens = model can burn 30s generating forever."""
		client = self._client()
		with mock.patch.object(client._session, "post",
				return_value=self._ok_response()) as post_mock:
			client.cleanup("raw", "moderate", "openai/gpt-oss-20b")
		body = post_mock.call_args.kwargs["json"]
		self.assertIn("max_completion_tokens", body)
		# Must be a positive integer cap; we pick 2000 by default.
		self.assertGreater(body["max_completion_tokens"], 0)
		self.assertLessEqual(body["max_completion_tokens"], 4096)

	def test_cleanup_omits_reasoning_from_response(self):
		"""We never look at the reasoning field; don't make Groq
		send the bytes back over the wire.
		"""
		client = self._client()
		with mock.patch.object(client._session, "post",
				return_value=self._ok_response()) as post_mock:
			client.cleanup("raw", "moderate", "openai/gpt-oss-20b")
		body = post_mock.call_args.kwargs["json"]
		self.assertEqual(body["include_reasoning"], False)

	def test_raw_mode_does_not_call_api(self):
		"""The raw mode is the explicit "no cleanup" branch. We must
		not even hit Groq, otherwise the user pays for nothing on
		every dictation where they've selected raw.
		"""
		client = self._client()
		with mock.patch.object(client._session, "post") as post_mock:
			result = client.cleanup("raw text", "raw", "openai/gpt-oss-20b")
		post_mock.assert_not_called()
		self.assertEqual(result, "raw text")

	def test_cleanup_logs_prompt_cache_hit_rate(self):
		"""The client should log cached_tokens / prompt_tokens so we
		can verify Groq's prompt caching is working. Without this
		visibility, a silent cache regression would be invisible.
		"""
		client = self._client()
		resp = self._ok_response("cleaned")
		resp.json.return_value["usage"] = {
			"prompt_tokens": 1000,
			"completion_tokens": 50,
			"prompt_tokens_details": {"cached_tokens": 950},
		}
		with mock.patch.object(client._session, "post", return_value=resp):
			with mock.patch("groq_client.log") as mock_log:
				client.cleanup("raw", "moderate", "openai/gpt-oss-20b")
		# The log call should include the cached token count.
		cache_log_calls = [
			c for c in mock_log.info.call_args_list
			if "cache" in str(c).lower()
		]
		self.assertTrue(cache_log_calls,
			msg="cleanup() should log prompt-cache hit rate")

	def test_prompt_order_preserved_for_caching(self):
		"""Groq prompt caching requires the static prefix to be at
		the start of the request body. Our system prompt is always
		first in the messages list, so the prefix is cacheable.
		This test pins the message order so a future refactor
		can't silently break the cache hit rate.
		"""
		client = self._client()
		with mock.patch.object(client._session, "post",
				return_value=self._ok_response()) as post_mock:
			client.cleanup("transcript text", "moderate", "openai/gpt-oss-20b")
		body = post_mock.call_args.kwargs["json"]
		messages = body["messages"]
		self.assertEqual(messages[0]["role"], "system")
		self.assertEqual(messages[1]["role"], "user")
		# The transcript should be in the user message
		self.assertIn("transcript text", messages[1]["content"])


class GeminiCleanupPostBodyTests(unittest.TestCase):
	"""Pin the wire shape of the Gemini cleanup POST."""

	def setUp(self) -> None:
		# Import the real client now that the test stubs are installed
		# by the module-level setup.
		import globalPlugins.groqVoiceDictation.gemini_client as gc  # noqa: WPS433
		self._gc = gc

	def _ok_response(self, content: str = "cleaned text") -> mock.MagicMock:
		resp = mock.MagicMock()
		resp.ok = True
		resp.json.return_value = {
			"candidates": [{
				"content": {"parts": [{"text": content}]},
			}],
		}
		return resp

	def test_gemini_cleanup_caps_max_output_tokens(self):
		client = self._gc.GeminiClient(api_key="test_key")
		with mock.patch.object(client._session, "post",
				return_value=self._ok_response()) as post_mock:
			client.cleanup("raw", "moderate", "gemini-2.5-flash")
		body = post_mock.call_args.kwargs["json"]
		self.assertIn("maxOutputTokens", body["generationConfig"])
		self.assertGreater(body["generationConfig"]["maxOutputTokens"], 0)

	def test_gemini_cleanup_uses_same_temperature(self):
		client = self._gc.GeminiClient(api_key="test_key")
		with mock.patch.object(client._session, "post",
				return_value=self._ok_response()) as post_mock:
			client.cleanup("raw", "moderate", "gemini-2.5-flash")
		body = post_mock.call_args.kwargs["json"]
		self.assertEqual(body["generationConfig"]["temperature"], 0.3)


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


class LooksSuspiciousTests(unittest.TestCase):
	"""Lock in the auto-retry heuristic for prompt-induced start-skipping."""

	def test_empty_is_not_suspicious(self):
		# Empty results are handled separately; the heuristic is
		# only called for non-empty text.
		self.assertFalse(_looks_suspicious(""))

	def test_long_result_is_not_suspicious(self):
		# 5+ words: even if it starts with a suspect opener, we
		# trust the model — a real dictation utterance like "So I
		# was thinking about the deployment" is fine.
		self.assertFalse(_looks_suspicious("So I was thinking about the deployment"))
		self.assertFalse(_looks_suspicious("You know what I mean right"))

	def test_short_suspect_opener_is_suspicious(self):
		# Single-word or two-word results that start with a common
		# opener are exactly the boostN "start-skipping" pattern.
		self.assertTrue(_looks_suspicious("So."))
		self.assertTrue(_looks_suspicious("So"))
		self.assertTrue(_looks_suspicious("You"))
		self.assertTrue(_looks_suspicious("Yes"))
		self.assertTrue(_looks_suspicious("Thank you"))
		self.assertTrue(_looks_suspicious("you know"))

	def test_short_non_suspect_is_not_suspicious(self):
		# Short results that are NOT common openers (e.g. proper
		# nouns, technical terms) are likely real — don't retry
		# those.
		self.assertFalse(_looks_suspicious("PostgreSQL"))
		self.assertFalse(_looks_suspicious("kubernetes"))
		self.assertFalse(_looks_suspicious("docker compose"))
		self.assertFalse(_looks_suspicious("TensorFlow"))

	def test_capitalisation_does_not_matter(self):
		# Case-insensitive matching on the opener.
		self.assertTrue(_looks_suspicious("SO"))
		self.assertTrue(_looks_suspicious("So, listen"))
		self.assertTrue(_looks_suspicious("OKAY"))

	def test_punctuation_does_not_matter(self):
		self.assertTrue(_looks_suspicious("So,"))
		self.assertTrue(_looks_suspicious("You?"))
		self.assertTrue(_looks_suspicious("Yes!"))


class FilterHallucinatedSegmentsCompressionRatioTests(unittest.TestCase):
	"""The new compression-ratio filter must catch repetition loops."""

	def test_high_compression_ratio_is_dropped(self):
		segments = [
			{
				"text": "the the the the the the the the",
				"no_speech_prob": 0.0,
				"avg_logprob": -0.3,
				"compression_ratio": 3.5,  # above HARD_REJECT
			},
		]
		kept = _filter_hallucinated_segments(segments)
		self.assertEqual(kept, [])

	def test_borderline_compression_with_low_logprob_is_dropped(self):
		segments = [
			{
				"text": "I think I think I think",
				"no_speech_prob": 0.1,
				"avg_logprob": -0.9,  # below -0.7
				"compression_ratio": 2.5,  # above THRESHOLD
			},
		]
		kept = _filter_hallucinated_segments(segments)
		self.assertEqual(kept, [])

	def test_borderline_compression_with_high_logprob_is_kept(self):
		# Legitimate repetitive phrasing with high confidence is
		# kept — "yes yes yes" is a real answer the user gave.
		segments = [
			{
				"text": "yes yes yes",
				"no_speech_prob": 0.05,
				"avg_logprob": -0.2,
				"compression_ratio": 2.5,
			},
		]
		kept = _filter_hallucinated_segments(segments)
		self.assertEqual(kept, ["yes yes yes"])

	def test_low_compression_ratio_is_kept(self):
		segments = [
			{
				"text": "the quick brown fox",
				"no_speech_prob": 0.0,
				"avg_logprob": -0.2,
				"compression_ratio": 1.4,
			},
		]
		kept = _filter_hallucinated_segments(segments)
		self.assertEqual(kept, ["the quick brown fox"])

	def test_missing_compression_ratio_defaults_to_zero(self):
		# Old payloads without compression_ratio must not crash.
		segments = [
			{
				"text": "hello world",
				"no_speech_prob": 0.0,
				"avg_logprob": -0.2,
				# no compression_ratio key
			},
		]
		kept = _filter_hallucinated_segments(segments)
		self.assertEqual(kept, ["hello world"])


class TranscribeWithRetryTests(unittest.TestCase):
	"""The auto-retry path covers prompt-induced start-skipping."""

	def setUp(self) -> None:
		# transcribe() opens the WAV path to send it to the API, so
		# the tests need a real (empty) file on disk. We clean up
		# in tearDown.
		self._tmp = tempfile.NamedTemporaryFile(
			delete=False, suffix=".wav",
		)
		self._tmp.close()
		# The file must be readable; an empty file is fine because
		# the tests mock the HTTP layer.
		self.wav_path = self._tmp.name

	def tearDown(self) -> None:
		import os
		if os.path.exists(self.wav_path):
			os.unlink(self.wav_path)

	def _client(self) -> GroqClient:
		return GroqClient(api_key="test_key")

	def _ok_response(self, text: str) -> mock.MagicMock:
		resp = mock.MagicMock()
		resp.ok = True
		resp.json.return_value = {"text": text, "segments": []}
		return resp

	def _segments_response(self, text: str, **seg_kwargs) -> mock.MagicMock:
		resp = mock.MagicMock()
		resp.ok = True
		resp.json.return_value = {
			"text": text,
			"segments": [
				{
					"text": text,
					"no_speech_prob": 0.0,
					"avg_logprob": -0.2,
					"compression_ratio": 1.4,
					**seg_kwargs,
				},
			],
		}
		return resp

	def test_long_first_pass_skips_retry(self):
		client = self._client()
		first = "This is a perfectly fine long dictation result"
		with mock.patch.object(client._session, "post",
				side_effect=[self._ok_response(first)]) as post_mock:
			result = client.transcribe_with_retry(
				self.wav_path, "whisper-large-v3-turbo",
				prompt="Python, JavaScript", auto_retry=True,
			)
		self.assertEqual(result, first)
		self.assertEqual(post_mock.call_count, 1)

	def test_suspicious_first_pass_triggers_retry(self):
		client = self._client()
		with mock.patch.object(client._session, "post",
				side_effect=[
					self._ok_response("So."),  # suspicious
					self._ok_response("So I was going to the store"),  # retry
				]) as post_mock:
			result = client.transcribe_with_retry(
				self.wav_path, "whisper-large-v3-turbo",
				prompt="Python, JavaScript", auto_retry=True,
			)
		self.assertEqual(result, "So I was going to the store")
		self.assertEqual(post_mock.call_count, 2)
		# The retry must drop the prompt.
		retry_call = post_mock.call_args_list[1]
		form = retry_call.kwargs.get("data") or retry_call[1].get("data")
		self.assertNotIn("prompt", form)

	def test_auto_retry_disabled_skips_retry(self):
		client = self._client()
		with mock.patch.object(client._session, "post",
				side_effect=[self._ok_response("So.")]) as post_mock:
			result = client.transcribe_with_retry(
				self.wav_path, "whisper-large-v3-turbo",
				prompt="x", auto_retry=False,
			)
		self.assertEqual(result, "So.")
		self.assertEqual(post_mock.call_count, 1)

	def test_retry_also_suspicious_returns_longer_one(self):
		# Both passes are bad; we return whichever has more words
		# rather than always preferring the first.
		client = self._client()
		with mock.patch.object(client._session, "post",
				side_effect=[
					self._ok_response("So I was"),  # suspicious (4 words, "so")
					self._ok_response("So we are"),  # suspicious (4 words, "so")
				]) as post_mock:
			result = client.transcribe_with_retry(
				self.wav_path, "whisper-large-v3-turbo",
				prompt="x", auto_retry=True,
			)
		# Both have 4 words; either is acceptable — the contract is
		# "return the longer one, ties go to first".
		self.assertEqual(result, "So I was")
		self.assertEqual(post_mock.call_count, 2)

	def test_retry_empty_returns_first(self):
		# If the retry returns empty (model gave up entirely), the
		# first pass is the better answer.
		client = self._client()
		with mock.patch.object(client._session, "post",
				side_effect=[
					self._ok_response("So."),
					self._ok_response(""),
				]) as post_mock:
			result = client.transcribe_with_retry(
				self.wav_path, "whisper-large-v3-turbo",
				prompt="x", auto_retry=True,
			)
		self.assertEqual(result, "So.")

	def test_retry_keeps_language_param(self):
		# Language is passed on both passes — it speeds up inference
		# and avoids language mis-detection.
		client = self._client()
		with mock.patch.object(client._session, "post",
				side_effect=[
					self._ok_response("So."),
					self._ok_response("So we are going"),
				]) as post_mock:
			client.transcribe_with_retry(
				self.wav_path, "whisper-large-v3-turbo",
				prompt="x", language="en", auto_retry=True,
			)
		for call in post_mock.call_args_list:
			form = call.kwargs.get("data") or call[1].get("data")
			self.assertEqual(form.get("language"), "en")


class GetAudioProcessingTests(unittest.TestCase):
	"""The new config helper must return all the audio-processing knobs."""

	def test_returns_defaults_for_empty_config(self):
		from config_manager import get_audio_processing
		conf = {}
		cfg = get_audio_processing(conf)
		self.assertEqual(cfg["preRollMs"], 0)
		self.assertEqual(cfg["preTrimSilenceMs"], 300)
		self.assertEqual(cfg["trailingTrimSilenceMs"], 300)
		self.assertTrue(cfg["autoRetryEnabled"])

	def test_returns_overrides_when_set(self):
		from config_manager import get_audio_processing
		conf = {
			"preRollMs": 500,
			"preTrimSilenceMs": 100,
			"trailingTrimSilenceMs": 200,
			"autoRetryEnabled": False,
		}
		cfg = get_audio_processing(conf)
		self.assertEqual(cfg["preRollMs"], 500)
		self.assertEqual(cfg["preTrimSilenceMs"], 100)
		self.assertEqual(cfg["trailingTrimSilenceMs"], 200)
		self.assertFalse(cfg["autoRetryEnabled"])

	def test_confspec_contains_new_keys(self):
		for key in (
			"preRollMs", "preTrimSilenceMs", "trailingTrimSilenceMs",
			"autoRetryEnabled",
		):
			self.assertIn(key, config_manager.CONFSPEC, f"missing CONFSPEC key: {key}")

	def test_default_prompts_are_shorter(self):
		# The trimmed default prompt slots should each fit well
		# under the 224-token Whisper limit. We don't tokenise here
		# but we can check that no default slot has more than 200
		# characters, which is a rough proxy.
		from config_manager import DEFAULT_PROMPT_SLOTS
		for index, slot in enumerate(DEFAULT_PROMPT_SLOTS):
			if slot:
				self.assertLess(
					len(slot), 200,
					msg=f"default prompt slot {index} is too long: {slot!r}",
				)


class GeminiModeratePromptParityTests(unittest.TestCase):
	"""The Gemini cleanup prompt must match the Groq cleanup prompt's
	ASR-mishearing license in shape and content. If one is updated the
	other should be too — these tests pin the parity.
	"""

	def _moderate(self) -> str:
		return gemini_client._gemini_cleanup_system_prompt("moderate")

	def test_gemini_moderate_licenses_asr_mishearing_fixes(self):
		content = self._moderate()
		self.assertIn("ASR mishearings", content)
		for phrase in (
			"Pronoun mishears",
			"Homophones that change meaning",
			"Compound terms and proper nouns",
		):
			self.assertIn(phrase, content, msg=f"gemini moderate missing {phrase!r}")

	def test_gemini_moderate_examples_match_groq(self):
		"""The same concrete examples should appear in both prompts so
		the model learns the same error patterns regardless of provider.
		"""
		gemini_content = self._moderate()
		groq_content = build_cleanup_messages("anything goes", "moderate")[0]["content"]
		# Examples the user actually asked for (pronoun mishear) plus
		# a sample of compound-term examples.
		for example in (
			"'tensor flow' -> 'TensorFlow'",
			"'API gate way' -> 'API gateway'",
			"'post gress SQL' -> 'PostgreSQL'",
		):
			self.assertIn(example, gemini_content, msg=f"gemini missing {example!r}")
			self.assertIn(example, groq_content, msg=f"groq missing {example!r}")

	def test_gemini_moderate_forbids_rephrasing(self):
		content = self._moderate()
		self.assertIn("paraphrase", content.lower())
		self.assertIn("rephrase", content.lower())

	def test_gemini_moderate_protects_short_utterances_from_asr_fixes(self):
		content = self._moderate()
		self.assertIn("under 8 words", content)
		self.assertIn("do NOT apply any ASR fix", content)

	def test_gemini_moderate_caps_asr_fixes(self):
		content = self._moderate()
		self.assertIn("5-10%", content)


if __name__ == "__main__":
	unittest.main()
