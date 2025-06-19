import argparse
import json
import tensorflow as tf
from phonema.model.conformer_model import build_phoneme_segmentation_model

def parse_args():
    p = argparse.ArgumentParser(description="Train phoneme segmentation model")
    p.add_argument("--train_npz", required=True)
    p.add_argument("--val_npz", required=True)
    p.add_argument("--phoneme_index", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output_model", default="model.h5")
    return p.parse_args()

def load_data(npz_path):
    data = tf.data.Dataset.load(npz_path)
    return data

def main():
    args = parse_args()
    with open(args.phoneme_index, "r") as f:
        phoneme_index = json.load(f)
    num_phonemes = len(phoneme_index)

    # Model
    model = build_phoneme_segmentation_model(num_phonemes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Data
    train_ds = load_data(args.train_npz)
    val_ds = load_data(args.val_npz)

    train_ds = train_ds.shuffle(1000).padded_batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.padded_batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Callback
    cb = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb)
    model.save(args.output_model)
    print(f"Model saved to {args.output_model}")

if __name__ == "__main__":
    main()

