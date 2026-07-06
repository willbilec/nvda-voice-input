"""Audio preprocessing utilities for dictation.

Pure functions that operate on raw int16 PCM frame bytes to:

* find the first / last sample that exceeds a silence threshold (used to
  locate speech in a recorded WAV),
* trim leading and trailing silence while keeping a small padding on each
  side so Whisper still has acoustic context, and
* compute RMS for noise-floor estimation.

The recorder writes a WAV that already includes a fixed lead-in silence;
this module can be applied after the recording stops so the file Whisper
receives is short, dense with speech, and starts/ends on a small pad of
silence rather than seconds of nothing.

All helpers accept either a single ``bytes`` blob or an iterable of frame
blobs (the shape the recorder keeps in its ``self._frames`` list) so they
can be used both on the raw frame buffer and on a re-loaded WAV.
"""
import array
import math
from typing import Iterable, Union

FramesLike = Union[bytes, bytearray, Iterable[Union[bytes, bytearray]]]


def _join_frames(frames: FramesLike) -> bytes:
	"""Return the concatenation of the input frames as a single ``bytes`` blob."""
	if isinstance(frames, (bytes, bytearray)):
		return bytes(frames)
	return b"".join(bytes(f) for f in frames)


def _to_samples(frames: FramesLike, sample_width: int = 2) -> array.array:
	"""Decode ``frames`` (int16 PCM) into an ``array.array`` of signed shorts."""
	joined = _join_frames(frames)
	if sample_width != 2:
		# The recorder is hard-coded to int16. Anything else is a programmer
		# error in this codebase; raise loudly rather than silently miscompute.
		raise ValueError(f"only int16 PCM is supported (sample_width={sample_width})")
	samples = array.array("h")
	samples.frombytes(joined)
	return samples


def frame_peak(frame: bytes) -> int:
	"""Peak amplitude of a single int16 PCM frame (absolute value, 0-32767)."""
	if not frame:
		return 0
	samples = array.array("h")
	samples.frombytes(frame)
	if not samples:
		return 0
	return max(abs(sample) for sample in samples)


def frame_rms(frame: bytes) -> float:
	"""RMS amplitude of a single int16 PCM frame, in the 0-32767 range."""
	if not frame:
		return 0.0
	samples = array.array("h")
	samples.frombytes(frame)
	if not samples:
		return 0.0
	sum_sq = 0
	for sample in samples:
		sum_sq += sample * sample
	return math.sqrt(sum_sq / len(samples))


def calculate_rms(frames: FramesLike) -> float:
	"""RMS amplitude of concatenated frames, in the 0-32767 range.

	Returns 0.0 for empty input. Used for adaptive noise-floor estimation
	when the configured silence threshold does not match the user's mic.
	"""
	joined = _join_frames(frames)
	if not joined:
		return 0.0
	return frame_rms(joined)


def find_first_voice_sample(frames: FramesLike, threshold: int) -> int:
	"""Sample offset of the first sample whose absolute value exceeds ``threshold``.

	Returns -1 if no such sample exists in the audio. The returned offset is
	in samples (not bytes) and can be converted to milliseconds by dividing
	by the sample rate.
	"""
	joined = _join_frames(frames)
	if not joined:
		return -1
	samples = array.array("h")
	samples.frombytes(joined)
	for index, sample in enumerate(samples):
		if abs(sample) > threshold:
			return index
	return -1


def find_last_voice_sample(frames: FramesLike, threshold: int) -> int:
	"""Sample offset *just past* the last sample whose absolute value exceeds ``threshold``.

	Returns -1 if no such sample exists. The returned offset is exclusive
	(so it can be used directly as a slice end) and is in samples.
	"""
	joined = _join_frames(frames)
	if not joined:
		return -1
	samples = array.array("h")
	samples.frombytes(joined)
	for index in range(len(samples) - 1, -1, -1):
		if abs(samples[index]) > threshold:
			return index + 1
	return -1


def has_voice(frames: FramesLike, threshold: int = 200) -> bool:
	"""Cheap check: does any sample in the audio exceed ``threshold``?

	Used by the recorder to gate the silence detection callback on real
	speech rather than ambient noise. A separate threshold from the main
	silence detection is intentional — the silence detection threshold is
	chosen to be above the noise floor, this one is meant to catch even
	quiet speech onset.
	"""
	return find_first_voice_sample(frames, threshold) >= 0


def trim_silence(
	frames: FramesLike,
	rate: int,
	threshold: int,
	leading_pad_ms: int = 300,
	trailing_pad_ms: int = 300,
	sample_width: int = 2,
) -> bytes:
	"""Trim leading and trailing silence, keeping a small padding on each side.

	The padding is in milliseconds of audio kept before the first voice
	sample and after the last voice sample. A non-zero padding is important:
	Whisper performs better when the speech is not sliced exactly at a
	phoneme boundary, and a small leading silence gives the model a few
	frames of context so the very first word is not lost.

	Returns the trimmed audio as raw int16 PCM bytes. Returns an empty
	``bytes`` if the input contains no audio above ``threshold`` (the
	caller should treat this as "no speech detected").
	"""
	if rate <= 0:
		raise ValueError(f"rate must be positive (got {rate})")
	samples = _to_samples(frames, sample_width=sample_width)
	total = len(samples)
	if total == 0:
		return b""

	first_voice = -1
	for index, sample in enumerate(samples):
		if abs(sample) > threshold:
			first_voice = index
			break
	if first_voice < 0:
		return b""

	last_voice = total
	for index in range(total - 1, -1, -1):
		if abs(samples[index]) > threshold:
			last_voice = index + 1
			break

	leading_pad = int((leading_pad_ms / 1000.0) * rate)
	trailing_pad = int((trailing_pad_ms / 1000.0) * rate)
	start = max(0, first_voice - leading_pad)
	end = min(total, last_voice + trailing_pad)
	if end <= start:
		return b""

	return samples[start:end].tobytes()


def estimate_noise_floor(
	frames: FramesLike,
	head_only: bool = True,
	sample_width: int = 2,
) -> float:
	"""Estimate the ambient noise floor of the audio as an RMS value.

	By default this looks only at the first 500ms of the audio (the
	assumption being that the speaker paused briefly before starting to
	talk). Pass ``head_only=False`` to estimate from the whole recording
	instead — useful when silence detection already stripped the leading
	quiet section and the noise floor is what is left.

	The returned value is in the 0-32767 range. A reasonable silence
	detection threshold is roughly 3-4x the noise floor; that multiplier
	is a deliberate choice for the caller, not a constant here.
	"""
	joined = _join_frames(frames)
	if not joined:
		return 0.0
	if head_only and len(joined) > 0:
		# Cap the inspection window at 500ms of int16 mono. The recorder
		# emits int16 mono, so 2 bytes per sample is correct.
		sample_rate_for_cap = 16000  # only used to size the cap window
		max_bytes = sample_rate_for_cap * 2 * 2  # 2 seconds * 2 bytes * 2 channels worth, but we are mono
		# A simpler cap: first 16KB ~= 0.5s at 16kHz mono int16
		max_bytes = 16000
		if len(joined) > max_bytes:
			joined = joined[:max_bytes]
	return frame_rms(joined)
