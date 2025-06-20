import argparse
import os
import tensorflow as tf
from utils.config import load_config
from utils.index_dataset import create_tf_dataset_from_index as create_dataset
from phonema.model.conformer_model import build_phoneme_segmentation_model as build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--train_index", type=str, default="output/train_index.csv")
    parser.add_argument("--dev_index", type=str, default="output/dev_index.csv")
    parser.add_argument("--train_mel_dir", type=str, required=True)
    parser.add_argument("--train_target_dir", type=str, required=True)
    parser.add_argument("--dev_mel_dir", type=str, required=True)
    parser.add_argument("--dev_target_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)

    print("📦 Caricamento dataset...")
    train_dataset = create_dataset(
        args.train_index, args.train_mel_dir, args.train_target_dir,
        batch_size=args.batch_size, shuffle=False
    )
    dev_dataset = create_dataset(
        args.dev_index, args.dev_mel_dir, args.dev_target_dir,
        batch_size=args.batch_size, shuffle=False
    )

    print("🧠 Creazione modello...")
    model = build_model(
        n_mels=config["n_mels"],
        n_classes=config["num_classes"],
        dropout=config.get("dropout", 0.3)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.get("lr", 1e-3)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # CALLBACKS
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "epoch_{epoch:02d}_val{val_loss:.4f}.keras"),
            save_weights_only=False,
            save_best_only=False
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output_dir, "best_model.keras"),
            monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(args.output_dir, "training_log.csv"))
    ]

    print("🚀 Inizio training...")
    model.fit(
        train_dataset,
        validation_data=dev_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    print("💾 Salvataggio modello finale...")
    model.save(os.path.join(args.output_dir, "final_model.keras"))

if __name__ == "__main__":
    main()

