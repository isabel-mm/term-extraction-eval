import os
import re
import random
import spacy
import nltk
from nltk.corpus import stopwords

# Descargar stopwords de NLTK si no están descargadas
nltk.download("stopwords")

# Cargar modelo de spaCy en inglés
nlp = spacy.load("en_core_web_trf")

# Lista de stopwords de NLTK y spaCy, excluyendo 'of'
stop_words = set(stopwords.words("english")) | set(nlp.Defaults.stop_words)
stop_words.discard("of")  # Asegurar que 'of' no se elimine

# Nombre del archivo de la stoplist académica
STOPLIST_FILE = "academica.txt"

# Directorios
INPUT_FOLDER = "original_texts"
OUTPUT_FOLDER = "processed_texts"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---- (1) Cargar la stoplist académica ----
def load_academic_stoplist(stoplist_file):
    """Carga la stoplist académica desde un archivo de texto."""
    if os.path.exists(stoplist_file):
        with open(stoplist_file, "r", encoding="utf-8") as file:
            stoplist = [line.strip().lower() for line in file if line.strip()]
        return sorted(set(stoplist), key=len, reverse=True)
    return []

academic_stoplist = load_academic_stoplist(STOPLIST_FILE)

# ---- (2) Eliminar frases académicas antes de cualquier otro procesamiento ----
def remove_academic_phrases(text, stoplist):
    """Elimina expresiones académicas completas antes de la tokenización."""
    for phrase in stoplist:
        phrase_pattern = r'(?i)(?<!\w)' + re.escape(phrase) + r'(?!\w)'
        text = re.sub(phrase_pattern, ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ---- (3) Eliminar caracteres no alfabéticos ----
def remove_non_alphabetic(text):
    """Elimina todo excepto caracteres alfabéticos y espacios."""
    return re.sub(r'[^a-zA-Z\s]', ' ', text)

# ---- (4) Procesar el texto ----
def process_text(text):
    """Primero elimina frases académicas, luego tokeniza, lematiza y elimina stopwords."""
    text = text.lower()
    
    # Primero eliminar frases académicas
    text = remove_academic_phrases(text, academic_stoplist)
    
    # Eliminar caracteres no alfabéticos
    text = remove_non_alphabetic(text)
    
    # Procesar con spaCy
    nlp.max_length = max(len(text), 1000000)
    doc = nlp(text)
    
    # Aplicar lematización solo a plurales y quitar stopwords (excepto 'of')
    processed_tokens = [
        token.lemma_ if token.tag_ in {"NNS", "NNPS"} else token.text
        for token in doc if token.text.lower() in {"of"} or token.text.lower() not in stop_words
    ]
    
    return " ".join(processed_tokens)

# ---- (5) Procesar todos los archivos en la carpeta ----
def process_corpus(input_folder, output_folder):
    """Procesa todos los archivos .txt en la carpeta de entrada y guarda los resultados en la carpeta de salida."""
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt") and file_name != STOPLIST_FILE:
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Leer contenido del archivo
            with open(input_path, "r", encoding="utf-8") as file:
                raw_text = file.read()

            # Procesar texto
            processed_text = process_text(raw_text)

            # Guardar el texto procesado
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(processed_text)

            print(f"✅ Procesado: {file_name} → {output_path}")

# ---- (6) Ejecutar el procesamiento ----
process_corpus(INPUT_FOLDER, OUTPUT_FOLDER)

print("\n✅ Todos los archivos han sido procesados y guardados en 'processed_texts'.")
