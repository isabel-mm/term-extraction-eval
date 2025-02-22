import os
import spacy
import nltk

# Descargar recursos de NLTK
nltk.download("punkt")

# Cargar modelo de spaCy en inglés
nlp = spacy.load("en_core_web_trf")

# Directorios y archivos
ORIGINAL_FOLDER = "original_texts"  # Carpeta con los archivos .txt originales
PROCESSED_FOLDER = "processed_texts"  # Carpeta con los textos procesados
OUTPUT_ORIGINAL = "corpus_completo_original.txt"
OUTPUT_PROCESSED = "corpus_completo_procesado.txt"
OUTPUT_MD = "tabla.md"

def merge_texts(folder, output_file):
    """Concatena todos los archivos .txt en un único archivo."""
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_name in sorted(os.listdir(folder)):
            if file_name.endswith(".txt"):
                file_path = os.path.join(folder, file_name)
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n")

    print(f"\n✅ Corpus combinado guardado en {output_file}")

def get_text_stats(text):
    """Procesa el texto en fragmentos y calcula métricas textuales."""
    chunk_size = 100_000  # Procesar en fragmentos de 100,000 caracteres
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    total_tokens, total_words, total_types, total_lexical = 0, 0, set(), 0

    for chunk in chunks:
        doc = nlp(chunk)  # Procesar fragmento

        tokens = [token.text for token in doc]
        words = [token.text for token in doc if token.is_alpha]
        lexical_words = [token.text for token in doc if token.pos_ in {"NOUN", "ADJ", "VERB", "ADV"}]

        total_tokens += len(tokens)
        total_words += len(words)
        total_types.update(words)  # Se actualiza con palabras únicas
        total_lexical += len(lexical_words)

    # Type-Token Ratio (TTR) en porcentaje
    ttr = (len(total_types) / total_words * 100) if total_words > 0 else 0

    # Densidad léxica en porcentaje
    lexical_density = (total_lexical / total_words * 100) if total_words > 0 else 0

    return total_tokens, total_words, ttr, lexical_density

def analyze_corpus(file_path):
    """Lee un archivo de texto y calcula sus métricas."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return get_text_stats(text)

def save_to_markdown(original_stats, processed_stats, output_file):
    """Guarda las estadísticas en un archivo Markdown."""
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("# 📊 Comparación Estadística del Corpus (Texto Completo)\n\n")
        file.write("| Métrica                 | Corpus Original | Corpus Procesado |\n")
        file.write("|-------------------------|----------------|------------------|\n")
        file.write(f"| **Tokens totales**      | {original_stats[0]:,.0f} | {processed_stats[0]:,.0f} |\n")
        file.write(f"| **Palabras totales**    | {original_stats[1]:,.0f} | {processed_stats[1]:,.0f} |\n")
        file.write(f"| **TTR (%)**             | {original_stats[2]:.2f}% | {processed_stats[2]:.2f}% |\n")
        file.write(f"| **Densidad Léxica (%)** | {original_stats[3]:.2f}% | {processed_stats[3]:.2f}% |\n")

    print(f"\n✅ Resultados guardados en {output_file}")

# Generar archivos combinados
merge_texts(ORIGINAL_FOLDER, OUTPUT_ORIGINAL)
merge_texts(PROCESSED_FOLDER, OUTPUT_PROCESSED)

# Analizar corpus original y procesado
original_stats = analyze_corpus(OUTPUT_ORIGINAL)
processed_stats = analyze_corpus(OUTPUT_PROCESSED)

# Guardar resultados en Markdown
save_to_markdown(original_stats, processed_stats, OUTPUT_MD)
