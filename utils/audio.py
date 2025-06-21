import numpy as np
import librosa


def chunk_to_mel_frames(y, sr, n_fft, hop_length, n_mels):
    """Convert a mono audio chunk to mel-spectrogram frames.

    Parameters
    ----------
    y : ndarray
        1D array of audio samples.
    sr : int
        Sample rate of ``y``.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length in samples.
    n_mels : int
        Number of mel bands.

    Returns
    -------
    np.ndarray
        Array of shape ``(T, n_mels)`` containing mel frames in dB scale.
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=False,
        power=1.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.T.astype(np.float32)
