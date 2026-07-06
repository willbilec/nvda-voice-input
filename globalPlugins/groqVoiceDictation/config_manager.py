import json as _json

import config


SECTION = "groqVoiceDictation"

TRANSCRIPTION_PROVIDERS = [
	("groq", "Groq (Whisper)"),
]

TRANSCRIPTION_MODELS = [
	"whisper-large-v3-turbo",
	"whisper-large-v3",
]

GEMINI_CLEANUP_MODELS = [
	"gemini-2.5-flash-lite",
	"gemini-2.5-flash",
	"gemini-3.5-flash",
]

GEMINI_CLEANUP_RATE_LIMITS: dict[str, str] = {
	"gemini-2.5-flash-lite": "1,000",
	"gemini-2.5-flash": "250",
	"gemini-3.5-flash": "~250",
}

LANGUAGE_CHOICES = [
	("", "Auto-detect"),
	("en", "English"),
	("es", "Spanish"),
	("fr", "French"),
	("de", "German"),
	("it", "Italian"),
	("pt", "Portuguese"),
	("nl", "Dutch"),
	("ja", "Japanese"),
	("ko", "Korean"),
	("zh", "Chinese"),
	("ru", "Russian"),
	("ar", "Arabic"),
	("hi", "Hindi"),
]

CLEANUP_MODELS = [
	"openai/gpt-oss-20b",
	"openai/gpt-oss-120b",
	"llama-3.1-8b-instant",
	"llama-3.3-70b-versatile",
	"meta-llama/llama-4-scout-17b-16e-instruct",
	"qwen/qwen3-32b",
]

LLAMA_MODELS: set[str] = {
	"llama-3.1-8b-instant",
	"llama-3.3-70b-versatile",
	"meta-llama/llama-4-scout-17b-16e-instruct",
}

CLEANUP_MODES = [
	("raw", "Raw transcript"),
	("light", "Light cleanup"),
	("moderate", "Moderate cleanup"),
	("heavy", "Heavy rewrite"),
]

FEEDBACK_MODES = [
	("speech", "Speech only"),
	("tones", "Tones only"),
	("both", "Speech and tones"),
]

READBACK_MODES = [
	("off", "Off"),
	("after", "Read back after insertion"),
	("confirm", "Read back and confirm before insertion"),
]

PROMPT_SLOT_COUNT = 10

# Default prompt slots. The comma-separated glossary format biases
# Whisper toward specific proper nouns and jargon — useful for getting
# "PostgreSQL" recognised rather than "post gress SQL" — but a long
# glossary also increases the chance of the prompt-induced
# start-skipping failure mode (boostN documented this with hard
# numbers: 22.6 words/sec normally vs 5.2 on the chopped recording
# when the glossary was the trigger). The defaults below are kept
# under ~80 tokens and avoid the high-frequency opener words that
# most often collide with user speech. The 224-token cap is the
# Whisper limit; the auto-retry path in groq_client.py covers the
# case where a custom prompt still trips start-skipping.
DEFAULT_PROMPT_SLOTS = [
	"Programming: Python, JavaScript, TypeScript, React, Node, SQL, Git, API, JSON, YAML, Docker, Linux, Windows",
	"Workflow: dictation, transcription, meeting, notes, email, message, documentation, summary, report, agenda",
	"Tech stack: PostgreSQL, MongoDB, Redis, Kubernetes, async, await, endpoint, webhook, deployment, refactor",
	"",
	"",
	"",
	"",
	"",
	"",
	"",
]

DEFAULT_PROMPT_SLOTS_JSON = _json.dumps(DEFAULT_PROMPT_SLOTS, ensure_ascii=False)

CONFSPEC = {
	"apiKey": 'string(default="")',
	"transcriptionProvider": 'string(default="groq")',
	"transcriptionModel": 'string(default="whisper-large-v3-turbo")',
	"transcriptionLanguage": 'string(default="en")',
	"geminiApiKey": 'string(default="")',
	"promptSlots": f'string(default=\'{DEFAULT_PROMPT_SLOTS_JSON}\')',
	"activePromptSlot": "integer(default=0,min=0,max=9)",
	"cleanupMode": 'string(default="light")',
	"cleanupModel": 'string(default="openai/gpt-oss-20b")',
	# Reasoning effort for gpt-oss cleanup models. Default "low" — the
	# cleanup prompt is fully rule-bound (Punctuation + ASR fix license
	# with explicit gates) so the model does not need extended chain-of-
	# thought to produce a good answer. "low" cuts the cleanup latency
	# roughly in half on Groq vs the model default. Only honored by
	# gpt-oss models; ignored by every other model. Users who hit a
	# hard case can bump to "medium" or "high" in the Cleanup dialog.
	"cleanupReasoningEffort": 'string(default="low")',
	"microphoneDevice": "integer(default=-1,min=-1,max=9999)",
	"fallbackMicrophoneDevice": "integer(default=-1,min=-1,max=9999)",
	"fallbackEnabled": "boolean(default=true)",
	"fallbackPreflightMs": "integer(default=800,min=300,max=3000)",
	"silenceDetection": "boolean(default=true)",
	"silenceTimeout": "integer(default=2,min=1,max=15)",
	"feedbackMode": 'string(default="both")',
	"allowPasteFallback": "boolean(default=true)",
	"silenceThreshold": "integer(default=1500,min=100,max=32767)",
	"readbackMode": 'string(default="off")',
	"confirmTimeout": "integer(default=5,min=2,max=15)",
	"speakRawTranscript": "boolean(default=false)",
	# Audio-processing knobs. See audio_processor.py and
	# AudioRecorder for what each does. preRollMs is 0 by default
	# to keep the existing behaviour (no extra toggle→listening
	# delay); users with slow mics (AirPods, USB headsets) can opt
	# in. preTrimSilenceMs and trailingTrimSilenceMs match the
	# AudioRecorder defaults (300ms each) so the recorder's
	# internal trim produces a tight WAV that still has a small
	# pad of silence on each side of the speech.
	"preRollMs": "integer(default=0,min=0,max=2000)",
	"preTrimSilenceMs": "integer(default=300,min=0,max=2000)",
	"trailingTrimSilenceMs": "integer(default=300,min=0,max=2000)",
	# Auto-retry: when the first transcription pass looks
	# suspicious (short result, suspect opener word) the add-on
	# silently retries without the prompt. Off would skip the
	# second API call, but the prompt-induced start-skipping
	# workaround is high-value, so it defaults on.
	"autoRetryEnabled": "boolean(default=true)",
}


def ensure_config_spec() -> None:
	config.conf.spec[SECTION] = CONFSPEC


def get() -> config.AggregatedSection:
	ensure_config_spec()
	return config.conf[SECTION]


def update_base_profile(values: dict) -> None:
	ensure_config_spec()
	for key, value in values.items():
		config.conf[SECTION][key] = value
	try:
		for key, value in values.items():
			config.conf.profiles[0][SECTION][key] = value
	except (KeyError, AttributeError):
		config.conf.profiles[0][SECTION] = values


def index_for_value(options: list[tuple[str, str]], value: str) -> int:
	for index, (option_value, _label) in enumerate(options):
		if option_value == value:
			return index
	return 0


def label_list(options: list[tuple[str, str]]) -> list[str]:
	return [label for _value, label in options]


def load_prompt_slots(conf: dict | config.AggregatedSection) -> list[str]:
	raw = conf.get("promptSlots", "") if isinstance(conf, dict) else conf["promptSlots"]
	try:
		slots = _json.loads(raw) if isinstance(raw, str) else list(raw)
	except (_json.JSONDecodeError, TypeError):
		slots = list(DEFAULT_PROMPT_SLOTS)
	while len(slots) < PROMPT_SLOT_COUNT:
		slots.append("")
	return slots[:PROMPT_SLOT_COUNT]


def get_active_prompt(conf: dict | config.AggregatedSection) -> str:
	slots = load_prompt_slots(conf)
	index = conf.get("activePromptSlot", 0) if isinstance(conf, dict) else int(conf.get("activePromptSlot", 0))
	if 0 <= index < len(slots):
		return slots[index]
	return ""


def get_audio_processing(conf: dict | config.AggregatedSection) -> dict:
	"""Return the audio-processing knobs as a plain dict.

	Centralises the defaulting logic so the recorder and the worker
	agree on what each value is, and so adding a new knob does not
	require touching every call site.
	"""

	def _get(key: str, default):
		if isinstance(conf, dict):
			return conf.get(key, default)
		return conf[key]

	return {
		"preRollMs": int(_get("preRollMs", 0)),
		"preTrimSilenceMs": int(_get("preTrimSilenceMs", 300)),
		"trailingTrimSilenceMs": int(_get("trailingTrimSilenceMs", 300)),
		"autoRetryEnabled": bool(_get("autoRetryEnabled", True)),
	}
