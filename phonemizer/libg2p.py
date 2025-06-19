import ctypes
import subprocess

lib = ctypes.CDLL("/usr/local/lib/libespeak-ng.so")

# Inizializza espeak-ng (esatto come in C)
AUDIO_OUTPUT_SYNCHRONOUS = 0
lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
lib.espeak_Initialize.restype = ctypes.c_int
lib.espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, None, 0)

# Firma della tua funzione
lib.espeak_G2P_WithSeparator.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.espeak_G2P_WithSeparator.restype = ctypes.c_char_p

def g2p_with_separator(text: str, sep: str = "|", lang: str = "it") -> str:
    text_bytes = text.encode("utf-8")
    sep_bytes = sep.encode("utf-8")
    lang_bytes = lang.encode("utf-8")
    result = lib.espeak_G2P_WithSeparator(text_bytes, sep_bytes, lang_bytes)
    return result.decode("utf-8")

def check_language_supported(lang: str) -> bool:
    try:
        result = subprocess.run(["espeak-ng", "--voices=" + lang], capture_output=True, text=True)
        return any(lang in line for line in result.stdout.splitlines())
    except Exception:
        return False

