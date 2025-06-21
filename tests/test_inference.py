
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import glob
import json
import argparse

from phonema.model.conformer_model import build_phoneme_segmentation_model
from utils.viterbi import viterbi_decode

# === Argomenti ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--mel_dir", type=str, required=True)
parser.add_argument("--target_dir", type=str, required=True)
parser.add_argument("--phoneme_map", type=str, required=True)
parser.add_argument("--viterbi_matrix", type=str, default=None)
parser.add_argument("--max_frames", type=int, default=1000)
args = parser.parse_args()

# === Caricamento mappa fonemi ===
with open(args.phoneme_map, "r") as f:
    phoneme_map = json.load(f)
    index_to_phoneme = {int(v): k for k, v in phoneme_map.items()}

# === Caricamento matrice di transizione Viterbi ===
viterbi_matrix = None
if args.viterbi_matrix and os.path.exists(args.viterbi_matrix):
    viterbi_matrix = np.load(args.viterbi_matrix)
    print(f"[✓] Matrice Viterbi caricata: {args.viterbi_matrix}")
else:
    print("[i] Inference senza Viterbi")

# === Caricamento modello ===
model = build_phoneme_segmentation_model(n_mels=80, n_classes=len(phoneme_map))
model.load_weights(args.model)

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

# === Caricamento dataset ===
mel_files = sorted(glob.glob(os.path.join(args.mel_dir, "*.npy")))
target_files = sorted(glob.glob(os.path.join(args.target_dir, "*.npy")))

assert len(mel_files) == len(target_files), "Mismatch tra mel e target"

# === Inference ===
total_distance = 0
total_length = 0

for mel_path, target_path in tqdm(zip(mel_files, target_files), total=len(mel_files)):
    mel, target = load_sample(mel_path, target_path)
    mel = mel[:args.max_frames]
    target = target[:args.max_frames]

    x = np.expand_dims(mel, axis=0)
    y_pred = model.predict(x, verbose=0)
    logits = y_pred[0]

    if viterbi_matrix is not None:
        pred_ids = viterbi_decode(logits, viterbi_matrix)
    else:
        pred_ids = np.argmax(logits, axis=-1)

    ref_seq = decode_sequence(target)
    hyp_seq = decode_sequence(pred_ids)

    dist = levenshtein(ref_seq, hyp_seq)
    total_distance += dist
    total_length += len(ref_seq)

wer = total_distance / total_length if total_length > 0 else 1.0
print(f"Phoneme WER (Levenshtein): {wer:.3f}")

