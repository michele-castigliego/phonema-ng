import argparse
import os
import json
import numpy as np
from tqdm import tqdm

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<SIL>": 1,
    "<START>": 2,
}

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def save_npy(path, array):
    np.save(path, array)

def get_output_path(out_dir, audio_path):
    audio_id = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(out_dir, f"{audio_id}.npy")

def build_phoneme_index(phonemized_data):
    index = dict(SPECIAL_TOKENS)
    current_id = 10  # Fonemi IPA iniziano da 10
    for sample in phonemized_data:
        for phoneme in sample["phonemes"]:
            if phoneme.startswith("<LANG:"):
                continue
            if phoneme not in index:
                index[phoneme] = current_id
                current_id += 1
    return index

def create_targets(sample, phoneme_to_index, total_frames):
    phonemes = [
        p for p in sample["phonemes"]
        if not p.startswith("<LANG:")
    ]
    n_phonemes = len(phonemes)
    frames_per_phoneme = total_frames // n_phonemes
    remainder = total_frames % n_phonemes

    targets = []
    for i, phoneme in enumerate(phonemes):
        count = frames_per_phoneme + (1 if i < remainder else 0)
        targets.extend([phoneme_to_index[phoneme]] * count)
    return np.array(targets, dtype=np.int16)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--mel_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    phonemized_data = list(load_jsonl(args.input_jsonl))
    phoneme_to_index = build_phoneme_index(phonemized_data)

    for sample in tqdm(phonemized_data, desc=f"Processing {args.output_dir}"):
        audio_path = sample["audio_path"]
        mel_path = get_output_path(args.mel_dir, audio_path)
        if not os.path.exists(mel_path.replace(".npy", ".npz")):
            continue
        mel = np.load(mel_path.replace(".npy", ".npz"))["mel"]
        total_frames = mel.shape[1]
        frame_targets = create_targets(sample, phoneme_to_index, total_frames)
        save_npy(get_output_path(args.output_dir, audio_path), frame_targets)

    with open(os.path.join(args.output_dir, "phoneme_to_index.json"), "w", encoding="utf-8") as f:
        json.dump(phoneme_to_index, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

