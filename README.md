
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

# Generazione trascrizione fonetica
PYTHONPATH=. python scripts/phonemize.py --tsv DataSet/cv-corpus-18.0-2024-06-14/it/train.tsv --out output/phonemized_train.jsonl

