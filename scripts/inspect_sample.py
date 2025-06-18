import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import pandas as pd

def plot_mel_with_phonemes(mel, phonemes, hop_length, sr, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 4))
    librosa.display.specshow(mel.T, sr=sr, hop_length=hop_length,
                              x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title or "Mel-spectrogram + fonemi")

    n_frames = mel.shape[0]
    duration = n_frames * hop_length / sr

    positions = np.linspace(0, duration, num=len(phonemes), endpoint=False)

    for x, label in zip(positions, phonemes):
        ax.text(x, mel.shape[1] + 2, label, rotation=90,
                fontsize=9, verticalalignment='bottom', color='black')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📷 Plot salvato in: {save_path}")
        plt.close()
    else:
        plt.show()

def main(args):
    df = pd.read_csv(args.index_csv)
    row = df[df['id'] == args.id]
    if row.empty:
        print(f"❌ ID '{args.id}' non trovato in {args.index_csv}")
        return

    mel_path = row.iloc[0]['mel_path']
    data = np.load(mel_path, allow_pickle=True)
    mel = data['mel']
    phonemes = data['phonemes'].tolist()

    print(f"🔹 ID: {args.id}")
    print(f"🔸 Mel shape: {mel.shape}")
    print(f"🔸 N fonemi: {len(phonemes)}")

    plot_mel_with_phonemes(
        mel,
        phonemes,
        args.hop_length,
        args.sr,
        title=f"ID: {args.id}",
        save_path=args.save
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ispeziona mel-spectrogramma e fonemi")
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--save", type=str, default=None, help="Path per salvare PNG (opzionale)")
    args = parser.parse_args()
    main(args)

