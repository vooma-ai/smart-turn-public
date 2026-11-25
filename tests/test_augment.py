import numpy as np
import pytest

from smart_turn.augment import SAMPLE_RATE, add_noise, augment_clip, band_limit, shift_pitch

SPEECH_END = int(2.7 * SAMPLE_RATE)


def speech_then_pause():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(3 * SAMPLE_RATE) * 0.1).astype(np.float32)
    audio[SPEECH_END:] = 0
    return audio


def last_loud_second(audio, threshold_db=-40, frame=320):
    usable = len(audio) // frame * frame
    energy = np.sqrt(np.mean(np.square(audio[:usable].reshape(-1, frame)), axis=1))
    loud = np.where(20 * np.log10(energy / energy.max() + 1e-12) > threshold_db)[0]
    return (loud[-1] + 1) * frame / SAMPLE_RATE


@pytest.mark.parametrize("change", [
    lambda a: shift_pitch(a, 1.5),
    lambda a: shift_pitch(a, -1.5),
    lambda a: band_limit(a, 400, 2800),
])
def test_keeps_length_and_speech_end(change):
    audio = speech_then_pause()
    changed = change(audio)
    assert len(changed) == len(audio)
    assert last_loud_second(changed) == pytest.approx(last_loud_second(audio), abs=0.021)


def test_noise_fills_the_pause_well_below_the_speech():
    audio = speech_then_pause()
    noisy = add_noise(audio, snr_db=10, exponent=1, rng=np.random.default_rng(3))
    # NOTE(neel): The pause sits at the noise floor, 10 dB under the speech,
    # so speech still visibly ends where it did.
    assert last_loud_second(noisy, threshold_db=-6) == pytest.approx(
        last_loud_second(audio), abs=0.021
    )


def test_augment_clip_keeps_length_and_dtype():
    audio = speech_then_pause()
    rng = np.random.default_rng(1)
    for _ in range(5):
        out = augment_clip(audio, rng, {"noise_p": 1, "band_limit_p": 1, "pitch_p": 1})
        assert len(out) == len(audio)
        assert out.dtype == np.float32


def test_augment_clip_with_nothing_enabled_returns_the_clip():
    audio = speech_then_pause()
    out = augment_clip(audio, np.random.default_rng(2), {"noise_p": 0, "band_limit_p": 0, "pitch_p": 0})
    np.testing.assert_array_equal(out, audio)
