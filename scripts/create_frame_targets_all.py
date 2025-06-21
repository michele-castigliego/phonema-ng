
import os
import json
import numpy as np
from tqdm import tqdm
from utils.config import load_config
from utils.phoneme_map import update_phoneme_mapping

def assign_targets_uniformly(num_frames, phonemes, phoneme_to_index):
    target = np.zeros(num_frames, dtype=np.int16)
    n = len(phonemes)
    if n == 0:
        return target
    segment_length = num_frames // n
    for i, ph in enumerate(phonemes):
        start = i * segment_length
        end = num_frames if i == n - 1 else (i + 1) * segment_length
        target[start:end] = phoneme_to_index.get(ph, 0)
    return target

def process_split(split, config):
    split_dir = os.path.join(config["mel_output_dir"], split)
    output_dir = os.path.join(config["frame_targets_dir"], split)
    jsonl_path = os.path.join(config["phonemized_dir"], f"phonemized_{split}.jsonl")

    os.makedirs(output_dir, exist_ok=True)

    # indicizza le righe jsonl per base name del file
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data_by_base = {
            os.path.splitext(os.path.basename(json.loads(line)["audio_path"]))[0]: json.loads(line)
            for line in f
        }

    for fname in tqdm(os.listdir(split_dir), desc=f"🧩 Processing {split}"):
        if not fname.endswith(".npz"):
            continue
        base = os.path.splitext(fname)[0]
        npz_path = os.path.join(split_dir, fname)
        if base not in data_by_base:
            continue

        mel = np.load(npz_path)["mel"]
        sample = data_by_base[base]
        phonemes = [ph for ph in sample["phonemes"] if not ph.startswith("<LANG:")]
        phoneme_to_index = update_phoneme_mapping(
            phonemes,
            config["phoneme_map_path"],
            config.get("special_tokens")
        )

        target = assign_targets_uniformly(mel.shape[0], phonemes, phoneme_to_index)
        out_path = os.path.join(output_dir, f"{base}.npy")
        np.save(out_path, target)

if __name__ == "__main__":
    config = load_config("config.yaml")
    for split in ["train", "dev", "test"]:
        process_split(split, config)

