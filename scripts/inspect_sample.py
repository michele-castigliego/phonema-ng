import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os
import json
import re

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def plot_mel_with_phonemes(mel, phonemes, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mel.T, aspect="auto", origin="lower", interpolation="none")

    ax.set_ylabel("Mel bands")
    ax.set_xlabel("Time frames")

    if title:
        ax.set_title(title)

    T = mel.shape[0]
    num_phonemes = len(phonemes)
    step = T / num_phonemes if num_phonemes > 0 else T

    for i, phon in enumerate(phonemes):
        center = int((i + 0.5) * step)
        if center < T:
            ax.text(center, mel.shape[1] + 2, phon, ha="center", va="bottom", fontsize=9, rotation=90, color="black")

    fig.colorbar(im, ax=ax, orientation='vertical')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Salvato: {save_path}")
        plt.close()
    else:
        plt.show()

def main(args):
    config = load_config()

    try:
        df = pd.read_csv(args.index_csv)
    except Exception as e:
        print(f"❌ Errore nella lettura del CSV: {e}")
        return

    row = df[df['id'] == args.id]
    if row.empty:
        print(f"❌ ID '{args.id}' non trovato in {args.index_csv}")
        return

    mel_path = row.iloc[0]['mel_path']
    raw_phonemes = row.iloc[0]['phonemes']

    try:
        raw_phonemes = re.sub(r'"{2,}', '"', raw_phonemes.strip())
        phonemes = json.loads(raw_phonemes)
    except Exception as e:
        print(f"❌ Errore parsing phonemes (JSON): {e}")
        print(f"🔎 Stringa malformata:\n{raw_phonemes}")
        return

    mel_data = np.load(mel_path, allow_pickle=True)
    mel = mel_data['mel']

    print(f"🔹 ID: {args.id}")
    print(f"🔸 Mel shape: {mel.shape}")
    print(f"🔸 N fonemi: {len(phonemes)}")

    plot_mel_with_phonemes(
        mel,
        phonemes,
        title=f"ID: {args.id}",
        save_path=args.save
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ispeziona mel-spectrogramma (da .npz) e fonemi (da CSV)")
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--save", type=str, default=None, help="Path per salvare PNG (opzionale)")
    args = parser.parse_args()
    main(args)
