import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Parametri audio
sr = 16000
n_fft = 400
hop_length = 160
n_mels = 80

# Percorso di test
test_audio_path = "DataSet/cv-corpus-18.0-2024-06-14/it/clips/common_voice_it_20057443.mp3"

# Caricamento e trimming automatico
y, _ = librosa.load(test_audio_path, sr=sr)
intervals = librosa.effects.split(y, top_db=25)
y_trimmed = np.concatenate([y[start:end] for start, end in intervals])

# Estrazione mel-spectrogramma
S = librosa.feature.melspectrogram(
    y=y_trimmed,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    power=1.0
)
S_db = librosa.power_to_db(S, ref=np.max)

print("Mel-spectrogram shape:", S_db.shape)

# Visualizza lo spettrogramma
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.title('Mel-Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig("tests/test_mel_output.png")
print("✅ Spettrogramma salvato in tests/test_mel_output.png")

