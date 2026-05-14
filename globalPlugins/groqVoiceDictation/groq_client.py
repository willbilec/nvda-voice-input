import json

from logHandler import log
import requests


TRANSCRIPT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
USER_AGENT = "GroqVoiceDictation/0.1.0"


class GroqClientError(RuntimeError):
	def __init__(self, category: str, message: str) -> None:
		super().__init__(message)
		self.category = category
		self.message = message


class GroqClient:
	def __init__(self, api_key: str, timeout: int = 60) -> None:
		self.api_key = normalize_api_key(api_key)
		self.timeout = timeout
		self._session = requests.Session()
		self._session.headers.update(
			{
				"Authorization": f"Bearer {self.api_key}",
				"User-Agent": USER_AGENT,
			}
		)

	def transcribe(self, wav_path: str, model: str) -> str:
		if not self.api_key:
			raise GroqClientError("auth", "Set a Groq API key in settings.")
		with open(wav_path, "rb") as file_handle:
			response = self._session.post(
				TRANSCRIPT_URL,
				data={
					"model": model,
					"response_format": "verbose_json",
				},
				files={
					"file": ("dictation.wav", file_handle, "audio/wav"),
				},
				timeout=self.timeout,
			)
		payload = self._parse_json_response(response, TRANSCRIPT_URL)
		text = payload.get("text", "")
		return text.strip()

	def cleanup(self, text: str, mode: str, model: str) -> str:
		if mode == "raw":
			return text
		response = self._session.post(
			CHAT_URL,
			json={
				"model": model,
				"temperature": 0.2,
				"messages": build_cleanup_messages(text=text, mode=mode),
			},
			timeout=self.timeout,
		)
		payload = self._parse_json_response(response, CHAT_URL)
		try:
			content = payload["choices"][0]["message"]["content"]
		except (KeyError, IndexError, TypeError) as exc:
			raise GroqClientError("api", "Groq returned an invalid cleanup response.") from exc
		return str(content).strip() or text

	def _parse_json_response(self, response: requests.Response, url: str) -> dict:
		if not response.ok:
			log.error("Groq HTTP error %s for %s body=%s", response.status_code, url, response.text[:1000])
			raise map_http_error(response.status_code, response.text)
		try:
			return response.json()
		except json.JSONDecodeError as exc:
			log.error("Groq returned invalid JSON for %s body=%s", url, response.text[:1000])
			raise GroqClientError("api", "Groq returned invalid JSON.") from exc


def build_cleanup_messages(text: str, mode: str) -> list[dict]:
	if mode == "heavy":
		system_prompt = (
			"You rewrite speech transcripts into polished text. "
			"Preserve intent, but you may rephrase aggressively for readability."
		)
	else:
		system_prompt = (
			"You clean up speech transcripts lightly. "
			"Preserve the user's wording and meaning while fixing casing, punctuation, spacing, and obvious filler artifacts."
		)
	return [
		{"role": "system", "content": system_prompt},
		{
			"role": "user",
			"content": (
				"Return only the cleaned text, with no explanation.\n\n"
				f"Transcript:\n{text}"
			),
		},
	]


def build_multipart_body(*args, **kwargs) -> bytes:
	raise NotImplementedError("Multipart body builder is no longer used; requests handles multipart encoding.")


def map_http_error(status: int, response_text: str) -> GroqClientError:
	message = None
	try:
		payload = json.loads(response_text)
		message = payload.get("error", {}).get("message") or payload.get("message")
	except Exception:
		message = None
	if status in (401, 403):
		return GroqClientError("auth", message or "Groq rejected the API key.")
	if status == 429:
		return GroqClientError("rate", message or "Groq rate limit reached.")
	if 500 <= status <= 599:
		return GroqClientError("server", message or "Groq server error.")
	return GroqClientError("api", message or f"Groq request failed with HTTP {status}.")


def normalize_api_key(api_key: str) -> str:
	return "".join(str(api_key).split())
