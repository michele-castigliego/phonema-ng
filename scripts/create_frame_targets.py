from utils.config import load_config
from utils.phoneme_map import update_phoneme_mapping
import argparse
import os
import json
import numpy as np
from tqdm import tqdm

# Solo SIL attivo per ora, ma riserviamo gli ID 1-9
SPECIAL_TOKENS = {
    "<SIL>": 0,
}

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def save_npy(path, array):
    np.save(path, array)

def get_audio_id(audio_path):
    return os.path.splitext(os.path.basename(audio_path))[0]

def get_mel_path(mel_dir, audio_path):
    return os.path.join(mel_dir, f"{get_audio_id(audio_path)}.npz")

def get_target_path(output_dir, audio_path):
    return os.path.join(output_dir, f"{get_audio_id(audio_path)}.npy")

def build_phoneme_index(phonemized_data):
    index = dict(SPECIAL_TOKENS)
    current_id = 10  # IPA symbols start at 10
    for sample in phonemized_data:
        for p in sample["phonemes"]:
            if p.startswith("<LANG:"):
                continue
            if p not in index:
                index[p] = current_id
                current_id += 1
    return index

def create_targets(sample, phoneme_to_index, total_frames):
    phonemes = [p for p in sample["phonemes"] if not p.startswith("<LANG:")]
    n = len(phonemes)

    if n == 0:
        return np.full(total_frames, SPECIAL_TOKENS["<SIL>"], dtype=np.int16)

    per_phoneme = total_frames // n
    remainder = total_frames % n

    targets = []
    for i, p in enumerate(phonemes):
        count = per_phoneme + (1 if i < remainder else 0)
        targets.extend([phoneme_to_index[p]] * count)

    return np.array(targets[:total_frames], dtype=np.int16)

def main():
    config = load_config("config.yaml")
    MAX_FRAMES = config.get("max_frames", 1000)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--mel_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    phonemized_data = list(load_jsonl(args.input_jsonl))
    phoneme_to_index = build_phoneme_index(phonemized_data)

    for sample in tqdm(phonemized_data, desc=f"Processing {args.output_dir}"):
        mel_path = get_mel_path(args.mel_dir, sample["audio_path"])
        if not os.path.exists(mel_path):
            continue

        mel = np.load(mel_path)["mel"]
        total_frames = mel.shape[0]

        targets = create_targets(sample, phoneme_to_index, total_frames)
        save_npy(get_target_path(args.output_dir, sample["audio_path"]), targets)

    with open(os.path.join(args.output_dir, "phoneme_to_index.json"), "w", encoding="utf-8") as f:
        json.dump(phoneme_to_index, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()


