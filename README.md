
#Crea e attiva l’ambiente virtuale
python3 -m venv phonema-env
source phonema-env/bin/activate

# Installa pacchetti
pip install -r pip-requirements.txt

# Compilazione versione forkata e corretta di espeak-ng
cd third_party/espeak-ng/
./autogen.sh
./configure
make
sudo make install

# Generazione trascrizione fonetica, esempio singolo
PYTHONPATH=. python scripts/phonemize.py --tsv DataSet/cv-corpus-18.0-2024-06-14/it/train.tsv --out output/phonemized_train.jsonl

# Generazione trascrizione fonetica per training, validation, test
PYTHONPATH=. python scripts/phonemize_all.py

# Generazione spettrogrammi, esempio singolo
python scripts/extract_mels.py \
  --input_jsonl output/phonemized_train.jsonl \
  --output_dir output/mel_segments/train \
  --index_csv output/mel_segments/train_index.csv

# Generazione di tutti gli spettrogrammi
python scripts/extract_mels_all.py


# Output
output/
├── mel_segments/
│   ├── train/
│   │   ├── common_voice_it_XXXX.npz
│   │   └── ...
│   ├── dev/
│   └── test/
├── train_index.csv
├── dev_index.csv
└── test_index.csv


# Ispezione
python scripts/inspect_sample.py \
  --index_csv output/mel_segments/train_index.csv \
  --id common_voice_it_20057445

python scripts/inspect_sample.py \
  --index_csv output/mel_segments/train_index.csv \
  --id common_voice_it_20057445 \
  --save output/plots/20057445.png


