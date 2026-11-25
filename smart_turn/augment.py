"""Random, label-preserving changes to a training clip.

Every change keeps the clip's length and where its speech ends, because the
trailing pause is one of the cues the model learns a turn end from. The clip
is augmented before it is front-padded to 8 s, so the padding stays silent as
it is in production.
"""

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

SAMPLE_RATE = 16000


def _rms(audio):
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def _speech_rms(audio, frame=320):
    """RMS of the loudest frames, so a long quiet tail doesn't read as speech."""
    usable = len(audio) // frame * frame
    if usable == 0:
        return _rms(audio)
    frames = audio[:usable].reshape(-1, frame)
    energies = np.sqrt(np.mean(np.square(frames), axis=1))
    return float(np.mean(np.sort(energies)[-max(1, len(energies) // 4):]) + 1e-12)


def colored_noise(length, exponent, rng):
    """Gaussian noise whose power falls off as 1/f**exponent: 0 white, 1 pink, 2 brown."""
    spectrum = np.fft.rfft(rng.standard_normal(length))
    freqs = np.fft.rfftfreq(length)
    freqs[0] = freqs[1] if length > 1 else 1.0
    noise = np.fft.irfft(spectrum / freqs ** (exponent / 2), n=length)
    return noise / _rms(noise)


def add_noise(audio, snr_db, exponent, rng):
    noise = colored_noise(len(audio), exponent, rng)
    noise_rms = _speech_rms(audio) / (10 ** (snr_db / 20))
    return audio + noise * noise_rms


def band_limit(audio, low_hz, high_hz, sample_rate=SAMPLE_RATE):
    """Zero-phase band-pass, like a different handset or codec."""
    sos = butter(2, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, audio)


def _frame_rms(audio, frame):
    padded = np.pad(audio, (0, -len(audio) % frame))
    return np.sqrt(np.mean(np.square(padded.reshape(-1, frame)), axis=1))


def match_envelope(shifted, original, frame=160):
    """Caps shifted's loudness at original's, frame by frame. The pitch shifter
    smears speech about 80 ms into the pause after it; this keeps the pause."""
    gain = np.clip(_frame_rms(original, frame) / (_frame_rms(shifted, frame) + 1e-9), 0, 1)
    centers = np.arange(len(gain)) * frame + frame / 2
    return shifted * np.interp(np.arange(len(shifted)), centers, gain)


def shift_pitch(audio, semitones, sample_rate=SAMPLE_RATE):
    shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)
    return match_envelope(shifted, audio)


DEFAULT_AUGMENT = {
    "noise_p": 0.5,
    "snr_db": [10.0, 30.0],
    "band_limit_p": 0.3,
    "pitch_p": 0.3,
    "pitch_semitones": 1.5,
}


def augment_clip(audio, rng, settings=None):
    """A randomly changed copy of audio, the same length as audio."""
    settings = {**DEFAULT_AUGMENT, **(settings or {})}
    out = np.asarray(audio, dtype=np.float32)
    if len(out) < SAMPLE_RATE // 10:
        return out
    if rng.random() < settings["pitch_p"]:
        limit = settings["pitch_semitones"]
        out = shift_pitch(out, rng.uniform(-limit, limit))
    if rng.random() < settings["band_limit_p"]:
        out = band_limit(out, rng.uniform(150, 400), rng.uniform(2800, 3800))
    if rng.random() < settings["noise_p"]:
        out = add_noise(out, rng.uniform(*settings["snr_db"]), rng.choice([0, 1, 2]), rng)
    return out.astype(np.float32)
