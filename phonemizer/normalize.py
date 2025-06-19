import re
import unicodedata

def normalize_text(text: str) -> str:
    # Rimuove virgolette e simboli tipografici
    text = re.sub(r"[\"'“”‘’«»]", "", text)

    # Rimuove altri segni di punteggiatura
    text = re.sub(r"[^\w\s]", "", text)

    # Converte in minuscolo
    text = text.lower()

    # Normalizzazione Unicode (es. lettere accentate)
    text = unicodedata.normalize("NFKC", text)

    # Elimina spazi multipli e spazi iniziali/finali
    text = re.sub(r"\s+", " ", text).strip()

    return text

