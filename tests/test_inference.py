import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import glob
import json

from phonema.model.conformer_model import build_phoneme_segmentation_model

# === CONFIG ===
CONFIG = {
    "mel_dir": "output/mel_segments/",
    "target_dir": "output/frame_targets/",
    "model_path": "output/models/best_model.h5",
    "phoneme_map_path": "output/phoneme_to_index.json",
    "max_frames": 1000,
    "batch_size": 1,
}

# === Load phoneme map ===
with open(CONFIG["phoneme_map_path"], "r") as f:
    phoneme_map = json.load(f)
    index_to_phoneme = {int(v): k for k, v in phoneme_map.items()}

# === Load model ===
model = build_phoneme_segmentation_model(n_mels=80, n_classes=len(phoneme_map))
model.load_weights(CONFIG["model_path"])

# === Utility ===
def load_sample(mel_path, target_path):
    mel = np.load(mel_path)
    target = np.load(target_path)
    return mel, target

def decode_sequence(seq):
    return [index_to_phoneme.get(int(i), "?") for i in seq if i >= 0]

def levenshtein(a, b):
    dp = np.zeros((len(a)+1, len(b)+1), dtype=int)
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + (0 if a[i-1] == b[j-1] else 1)
            )
    return dp[len(a)][len(b)]

# === Load dataset ===
mel_files = sorted(glob.glob(os.path.join(CONFIG["mel_dir"], "*.npy")))
target_files = sorted(glob.glob(os.path.join(CONFIG["target_dir"], "*.npy")))

assert len(mel_files) == len(target_files), "Mismatch between mel and target files"

# === Inference & Evaluation ===
total_distance = 0
total_length = 0

for mel_path, target_path in tqdm(zip(mel_files, target_files), total=len(mel_files)):
    mel, target = load_sample(mel_path, target_path)
    mel = mel[:CONFIG["max_frames"]]
    target = target[:CONFIG["max_frames"]]

    x = np.expand_dims(mel, axis=0)  # batch size 1
    y_pred = model.predict(x, verbose=0)
    y_pred_ids = np.argmax(y_pred[0], axis=-1)

    ref_seq = decode_sequence(target)
    hyp_seq = decode_sequence(y_pred_ids)

    distance = levenshtein(ref_seq, hyp_seq)
    total_distance += distance
    total_length += len(ref_seq)

wer = total_distance / total_length if total_length > 0 else 1.0
print(f"Phoneme WER (Levenshtein): {wer:.3f}")

