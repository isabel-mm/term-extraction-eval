import os
import re

def normalize_spaces(text):
    """Elimina espacios múltiples y deja un solo espacio entre palabras."""
    return re.sub(r'\s+', ' ', text).strip()

def process_txt_files():
    """Procesa todos los archivos .txt en la misma carpeta que el script y normaliza espacios."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Carpeta donde está el script

    for file_name in os.listdir(script_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(script_dir, file_name)

            # Leer el contenido del archivo
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Normalizar espacios
            normalized_text = normalize_spaces(text)

            # Sobrescribir el archivo con el texto normalizado
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(normalized_text)

            print(f"✅ Normalizado: {file_name}")

# Ejecutar la normalización en todos los archivos .txt de la carpeta
process_txt_files()

print("\n✅ Todos los archivos han sido normalizados.")
