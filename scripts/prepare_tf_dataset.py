from utils.config import load_config
import os
import numpy as np
import tensorflow as tf

def load_sample(mel_path, target_path):
    mel = np.load(mel_path)["mel"]  # (T, 80)
    target = np.load(target_path)   # (T,)
    return mel, target

def create_dataset(mel_dir, target_dir):
    config = load_config("config.yaml")
    MAX_FRAMES = config.get("max_frames", 1000)
    mel_files = [f for f in os.listdir(mel_dir) if f.endswith(".npz")]

    def gen():
        for f in mel_files:
            mel_path = os.path.join(mel_dir, f)
            target_path = os.path.join(target_dir, f.replace(".npz", ".npy"))
            if not os.path.exists(target_path):
                continue
            mel, target = load_sample(mel_path, target_path)
            yield mel, target

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 80), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int16),
        )

    # Filtro campioni troppo lunghi
    dataset = dataset.filter(lambda mel, tgt: tf.shape(mel)[0] <= MAX_FRAMES)
    dataset = dataset.batch(8, drop_remainder=True)
    dataset = dataset.prefetch(1)
    )

    # Filtro campioni troppo lunghi
    dataset = dataset.filter(lambda mel, tgt: tf.shape(mel)[0] <= MAX_FRAMES)
    dataset = dataset.batch(8, drop_remainder=True)
    dataset = dataset.prefetch(1)


