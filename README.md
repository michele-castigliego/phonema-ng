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

```
output/
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
```

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
  --batch_size 32 --epochs 50 --patience 5 \
  --causal
```

Usa l'opzione `--causal` per abilitare una configurazione compatibile con lo streaming.

Il file `training_log.csv` verrà generato nella directory di output con le metriche di ogni epoca.

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

L'eseguibile `extract_mels` accetta gli stessi parametri principali dello script Python.


python tests/verify_data_integrity.py \
  --mel_dir output/mel_segments/train \
  --target_dir output/frame_targets/train



python scripts/train.py \
  --config config.yaml \
  --train_index output/train_index.csv \
  --train_mel_dir output/mel_segments/train/ \
  --train_target_dir output/frame_targets/train/ \
  --dev_index output/dev_index.csv \
  --dev_mel_dir output/mel_segments/dev/ \
  --dev_target_dir output/frame_targets/dev/ \
  --output_dir output/models/run1/ \
  --batch_size 8 \
  --epochs 50 \
  --patience 5 \
  --reset-output


### 6. Streaming Inference (`stream_inference.py`)
```bash
python scripts/stream_inference.py \
  --model output/models/best_model.keras \
  --decode
```
Il modello deve essere stato addestrato con l'opzione `--causal` per garantire uno streaming corretto.
