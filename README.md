
# PHONEMA - Riconoscimento Fonemico Frame-Level

Pipeline completa per la fonemizzazione, conversione audio, estrazione di mel-spectrogrammi e generazione dei target fonemici frame-level a partire da Common Voice.

---

## 🔧 Requisiti

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install python3.12 python3.12-venv python3.12-dev
```

Il callback `CpuTemperatureMonitor` richiede la libreria `psutil`.
È inclusa nel file `pip-requirements-12.txt` ma può essere installata anche con
`pip install psutil`.

---

## 🧪 Ambienti Virtuali

| Script                           | Ambiente Virtuale  | Python |
|----------------------------------|---------------------|--------|
| scripts/phonemize.py             | phonema-env-py12    | 3.12   |
| scripts/convert_mp3_to_wav.py    | phonema-env-py12    | 3.12   |
| scripts/extract_mels.py          | phonema-env-py12    | 3.12   |
| scripts/create_frame_targets.py  | phonema-env-py11    | 3.11   |
| scripts/train.py                 | phonema-env-py11    | 3.11   |
| scripts/prepare_tf_dataset.py    | phonema-env-py11    | 3.11   |
| tests/test_g2p.py                | phonema-env-py12    | 3.12   |

```bash
# Python 3.11
python3.11 -m venv phonema-env-py11
source phonema-env-py11/bin/activate
export PYTHONPATH=.
pip install -r pip-requirements-11.txt

# Python 3.12
python3.12 -m venv phonema-env-py12
source phonema-env-py12/bin/activate
export PYTHONPATH=.
pip install -r pip-requirements-12.txt
```

---

## 📦 Installazione `espeak-ng` (fork modificato)

```bash
cd third_party/espeak-ng/
./autogen.sh
./configure
make
sudo make install
```

---

## ⚙️ File di configurazione

```yaml
# config.yaml

dataset_dir: DataSet/cv-corpus-18.0-2024-06-14/it
phonemized_dir: output/
wav_output_dir: output/wav_trimmed/
mel_output_dir: output/mel_segments/

sr: 16000
n_fft: 400
hop_length: 160
n_mels: 80
top_db: 30

phoneme_separator: "|"
max_frames: null
pad_mode: repeat

label_smoothing: 0.0
```

---

## 📁 Output directory

(output/)
├── phonemized_train.jsonl
├── phonemized_dev.jsonl
├── phonemized_test.jsonl
├── wav_trimmed/
│   ├── train/
│   ├── dev/
│   └── test/
├── mel_segments/
│   ├── train/
│   ├── dev/
│   └── test/
├── frame_targets/
│   ├── train/
│   ├── dev/
│   └── test/
├── train_index.csv
├── dev_index.csv
├── test_index.csv
├── models/
│   ├── checkpoint/
│   ├── training_log.csv

---

## 🧵 Pipeline di elaborazione

### 1. Fonemizzazione (`phonemize_all.py`)
```bash
python scripts/phonemize_all.py
```

### 2. Conversione MP3 → WAV + trimming (`convert_mp3_to_wav_all.py`)
```bash
python scripts/convert_mp3_to_wav_all.py --num_workers 8
```

### 3. Estrazione dei mel-spectrogrammi (`extract_mels_all.py`)
```bash
python scripts/extract_mels_all.py --num_workers 8
```

### 4. Generazione target fonemici (`create_frame_targets_all.py`)
```bash
python scripts/create_frame_targets_all.py
```

### 5. Dataset shuffle
```bash
python scripts/create_shuffled_index_all.py
```

### 6. Costruzione matrice di transizione Viterbi
```bash
python scripts/viterbi_matrix.py \
  --target_dir output/frame_targets \
  --lang it \
  --output_dir output \
  --num_classes 256
```

### 7. Test di inferenza
```bash
python scripts/test_inference.py \
  --model output/models/best_model.h5 \
  --mel_dir output/mel_segments/test \
  --target_dir output/frame_targets/test \
  --phoneme_map output/phoneme_to_index.json \
  --viterbi_matrix output/viterbi-matrix_it.npy
```

---

## 🧠 Training del modello

```bash
python scripts/train.py \
  --config config.yaml \
  --train_mel_dir output/mel_segments/train/ \
  --train_target_dir output/frame_targets/train/ \
  --dev_mel_dir output/mel_segments/dev/ \
  --dev_target_dir output/frame_targets/dev/ \
  --output_dir output/models/ \
  --batch_size 8 --epochs 50 --patience 5 \
  --reset-output \
  --causal
```

---

## 🔍 Ispezione visiva

```bash
python scripts/inspect_sample.py \
  --index_csv output/mel_segments/train_index.csv \
  --id common_voice_it_20057443 \
  --save output/plots/sample.png
```

---

## 📦 Backup progetto

```bash
tar --exclude='output' --exclude='phonema-env*' --exclude='third_party' --exclude='.git' -czvf phonema_project.tar.gz .
```

---

## 🧪 Versione C++ (sperimentale)

```bash
cd src
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DYAML_BUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install

cd ../..
mkdir build && cd build
cmake ..
make
```

---

## 📁 Utility

```
utils/
└── viterbi.py            # Decoder Viterbi
scripts/
└── build_transition_matrix.py  # Generatore matrice Viterbi
└── test_inference.py           # Script valutazione inferenza
```

