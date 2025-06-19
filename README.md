# Crea e attiva l’ambiente virtuale
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Riferimento ambienti
+-------------------------------+------------------------+---------+
|           Script             |     Ambiente Virtuale  | Python  |
+-------------------------------+------------------------+---------+
| scripts/phonemize.py         | phonema-env-py12       | 3.12    |
| scripts/extract_mels_all.py  | phonema-env-py12       | 3.12    |
| scripts/create_frame_targets.py | phonema-env-py11    | 3.11    |
| scripts/train_model.py (es.) | phonema-env-py11       | 3.11    |
| tests/test_mel_extraction.py | phonema-env-py12       | 3.12    |
| tests/test_g2p.py            | phonema-env-py12       | 3.12    |
+-------------------------------+------------------------+---------+

# Output
output/
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
└── test_index.csv


python3.11 -m venv phonema-env-py11
source phonema-env-py11/bin/activate
export PYTHONPATH=.
pip install --upgrade pip
# Installa pacchetti
pip install -r pip-requirements-11.txt


python3.12 -m venv phonema-env-py12
source phonema-env-py12/bin/activate
export PYTHONPATH=.
pip install --upgrade pip
# Installa pacchetti
pip install -r pip-requirements-12.txt



# Installa pacchetti
pip install -r pip-requirements.txt

# Compilazione versione forkata e corretta di espeak-ng
cd third_party/espeak-ng/
./autogen.sh
./configure
make
sudo make install

# Generazione trascrizione fonetica, esempio singolo
python phonemize_multilang.py \
  --tsv DataSet/cv-corpus-18.0-2024-06-14/it/train.tsv \
  --out output/phonemized_train.jsonl \
  --audio_dir DataSet/cv-corpus-18.0-2024-06-14/it/clips \
  --lang it

# Generazione trascrizione fonetica per training, validation, test
python scripts/phonemize_all.py \
  --dataset_dir DataSet/cv-corpus-18.0-2024-06-14/it \
  --output_dir output/ \
  --lang it

# Generazione spettrogrammi, esempio singolo
python scripts/extract_mels.py \
  --input_jsonl output/phonemized_train.jsonl \
  --output_dir output/mel_segments/train \
  --index_csv output/mel_segments/train_index.csv

# Generazione di tutti gli spettrogrammi
python scripts/extract_mels_all.py


# TEST Ispezione
python scripts/inspect_sample.py   --index_csv output/mel_segments/train_index.csv   --id common_voice_it_20057443   --save output/plots/sample.png


## Estrazione dei Mel-spectrogrammi

Gli script `extract_mels.py` e `extract_mels_all.py` generano i mel-spectrogrammi segmentati per fonema a partire dai file `.jsonl` prodotti dalla fonemizzazione.

### ⚙️ Parametri audio

I parametri principali (`sr`, `n_fft`, `hop_length`, `n_mels`, `top_db`, ecc.) sono definiti nel file `config.yaml`.

### 🔹 Estrazione per singolo file

```bash
python scripts/extract_mels.py \
  --input_jsonl output/phonemized_train.jsonl \
  --output_dir output/mel_segments/train \
  --index_csv output/mel_segments/train_index.csv \
  --preemphasis   # opzionale
  # --no_norm     # per disabilitare la normalizzazione

python scripts/extract_mels_all.py \
  --phonemized_dir output/ \
  --output_dir output/mel_segments/ \
  --preemphasis   # opzionale
  # --no_norm     # opzionale


#Lo script create_frame_targets.py crea una sequenza che indica quale suono è presente in ogni momento dell’audio, distinguendo tra parlato e silenzio.
python scripts/create_frame_targets.py \
  --index_csv output/mel_segments/train_index.csv \
  --input_dir output/mel_segments/train \
  --output_dir output/frame_targets/train \
  --config config.yaml

python scripts/create_frame_targets_all.py


#Archivio
tar --exclude='output' --exclude='phonema-env' --exclude='third_party' --exclude='.git' -czvf phonema_project.tar.gz .


