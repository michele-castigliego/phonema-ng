import tensorflow as tf
from tensorflow.keras import layers, models

def conformer_block(inputs, d_model, num_heads, ff_dim, dropout):
    # Feed-forward module
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x_ff = layers.Dense(ff_dim, activation="relu")(x)
    x_ff = layers.Dropout(dropout)(x_ff)
    x_ff = layers.Dense(d_model)(x_ff)
    x = layers.Add()([inputs, x_ff])

    # Multi-head self-attention
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.Add()([x, attn])

    # Convolution module
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    conv = layers.Conv1D(filters=d_model, kernel_size=31, padding="same", activation="relu")(x)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Dropout(dropout)(conv)
    x = layers.Add()([x, conv])

    return x

def build_phoneme_segmentation_model(
    num_phonemes,
    input_n_mels=80,
    d_model=256,
    num_heads=4,
    ff_dim=512,
    num_blocks=4,
    dropout=0.1
):
    inp = layers.Input(shape=(None, input_n_mels), name="mel_input")  # (T, 80)
    x = layers.Reshape((-1, input_n_mels, 1))(inp)

    # Initial Conv2D layers to extract local features
    x = layers.Conv2D(64, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(dropout)(x)

    x = layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(dropout)(x)

    # Flatten frequency axis
    T = tf.shape(x)[1]
    F = x.shape[2] * x.shape[3]
    x = layers.Reshape((T, F))(x)
    x = layers.Dense(d_model)(x)

    # Conformer blocks
    for _ in range(num_blocks):
        x = conformer_block(x, d_model, num_heads, ff_dim, dropout)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    out = layers.TimeDistributed(layers.Dense(num_phonemes, activation="softmax"), name="phoneme_logits")(x)

    model = models.Model(inputs=inp, outputs=out, name="ConformerPhonemeSeg")
    return model

