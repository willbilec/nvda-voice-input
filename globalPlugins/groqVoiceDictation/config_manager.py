import config


SECTION = "groqVoiceDictation"

TRANSCRIPTION_MODELS = [
	"whisper-large-v3-turbo",
	"whisper-large-v3",
]

CLEANUP_MODELS = [
	"llama-3.1-8b-instant",
	"llama-3.3-70b-versatile",
]

CLEANUP_MODES = [
	("raw", "Raw transcript"),
	("light", "Light cleanup"),
	("heavy", "Heavy rewrite"),
]

FEEDBACK_MODES = [
	("speech", "Speech only"),
	("tones", "Tones only"),
	("both", "Speech and tones"),
]

CONFSPEC = {
	"apiKey": 'string(default="")',
	"transcriptionModel": 'string(default="whisper-large-v3-turbo")',
	"cleanupMode": 'string(default="light")',
	"cleanupModel": 'string(default="llama-3.1-8b-instant")',
	"microphoneDevice": "integer(default=-1,min=-1,max=9999)",
	"silenceDetection": "boolean(default=true)",
	"silenceTimeout": "integer(default=2,min=1,max=15)",
	"feedbackMode": 'string(default="both")',
	"allowPasteFallback": "boolean(default=true)",
	"silenceThreshold": "integer(default=500,min=100,max=5000)",
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
