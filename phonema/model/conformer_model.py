import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.positional_encoding import PositionalEncoding

class StreamingConformerBlock(keras.layers.Layer):
    def __init__(self, d_model, ff_dim, num_heads, dropout=0.1, causal=True):
        super().__init__()
        self.ff_pre = keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])

        self.attn_norm = layers.LayerNormalization()
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        self.dropout_attn = layers.Dropout(dropout)

        self.conv_norm = layers.LayerNormalization()
        self.conv_proj = layers.Conv1D(filters=2 * d_model, kernel_size=1, activation="gelu")
        self.causal = causal
        self.depthwise_conv = layers.DepthwiseConv1D(
            kernel_size=15,
            padding="valid" if causal else "same"
        )
        self.batch_norm = layers.BatchNormalization()
        self.conv_out = layers.Conv1D(filters=d_model, kernel_size=1)
        self.dropout_conv = layers.Dropout(dropout)

        self.ff_post = keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])

    def call(self, x, cache=None, training=False):
        # Feed Forward Pre
        x_ff = self.ff_pre(x, training=training)

        # Multi-Head Attention (with optional causal mask)
        attn_input = self.attn_norm(x)
        attn_output = self.mha(attn_input, attn_input, use_causal_mask=self.causal)
        attn_output = self.dropout_attn(attn_output, training=training)
        x = x + 0.5 * x_ff + attn_output

        # Convolution Module
        conv = self.conv_norm(x)
        conv = self.conv_proj(conv)

        if self.causal:
            pad_len = 15 - 1  # kernel_size - 1
            conv = tf.pad(conv, [[0, 0], [pad_len, 0], [0, 0]])

        conv = self.depthwise_conv(conv)
        conv = self.batch_norm(conv, training=training)
        conv = tf.nn.swish(conv)
        conv = self.conv_out(conv)
        conv = self.dropout_conv(conv, training=training)

        x += conv

        # Feed Forward Post
        x += self.ff_post(x, training=training)
        return x


def build_streaming_conformer_model(
    n_mels=80,
    n_classes=128,
    d_model=256,
    num_heads=4,
    ff_dim=512,
    num_layers=4,
    dropout=0.1,
    causal=True
):
    inputs = keras.Input(shape=(None, n_mels), name="mel_spectrogram")
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(d_model)(x)

    for _ in range(num_layers):
        x = StreamingConformerBlock(d_model, ff_dim, num_heads, dropout, causal)(x)

    outputs = layers.Dense(n_classes, name="frame_logits")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="StreamingConformer")

# Alias per compatibilità con codice esistente
build_phoneme_segmentation_model = build_streaming_conformer_model

