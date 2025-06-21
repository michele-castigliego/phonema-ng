import argparse
import json
import queue

import numpy as np
import sounddevice as sd
from tensorflow import keras

from utils.config import load_config
from utils.audio import chunk_to_mel_frames


def microphone_stream(sr: int, chunk_samples: int):
    """Generator that yields audio chunks from the default microphone."""
    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=sr, channels=1, blocksize=chunk_samples,
                        dtype="float32", callback=callback):
        while True:
            yield q.get()


def load_phoneme_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {v: k for k, v in mapping.items()}


def main(args):
    config = load_config(args.config)
    model = keras.models.load_model(args.model)

    idx_to_phoneme = None
    if args.decode:
        idx_to_phoneme = load_phoneme_map(config["phoneme_map_path"])

    sr = config["sr"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    n_mels = config["n_mels"]

    chunk_samples = int(sr * args.chunk_duration)
    mel_window = []
    max_window = args.window
    processed_frames = 0
    audio_buffer = np.array([], dtype=np.float32)
    all_probs = []

    print("🎙️  Avvio streaming... (Ctrl+C per terminare)")
    try:
        for chunk in microphone_stream(sr, chunk_samples):
            audio_buffer = np.concatenate([audio_buffer, chunk])
            mel = chunk_to_mel_frames(audio_buffer, sr, n_fft, hop_length, n_mels)
            new_mel = mel[processed_frames:]
            processed_frames = mel.shape[0]
            audio_buffer = audio_buffer[processed_frames * hop_length:]

            if len(new_mel) == 0:
                continue

            mel_window.extend(new_mel)
            if len(mel_window) > max_window:
                mel_window = mel_window[-max_window:]

            window_arr = np.array(mel_window, dtype=np.float32)
            preds = model(window_arr[None, ...], training=False).numpy()[0]
            new_probs = preds[-len(new_mel):]
            all_probs.append(new_probs)

            if idx_to_phoneme:
                phones = [idx_to_phoneme.get(int(np.argmax(p)), "?") for p in new_probs]
                print(" ".join(phones))
    except KeyboardInterrupt:
        pass

    if args.save_probs:
        probs = np.concatenate(all_probs, axis=0)
        np.savez(args.save_probs, probs=probs)
        print(f"✅ Probabilità salvate in {args.save_probs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming inference from microphone")
    parser.add_argument("--model", required=True, help="Path to trained .keras model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--chunk-duration", type=float, default=0.5,
                        help="Audio chunk duration in seconds")
    parser.add_argument("--window", type=int, default=160,
                        help="Number of mel frames in the sliding window")
    parser.add_argument("--decode", action="store_true",
                        help="Decode predictions to phonemes")
    parser.add_argument("--save-probs", type=str, default=None,
                        help="Optional path to save raw probabilities as .npz")
    args = parser.parse_args()
    main(args)

