import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Definir rutas
documentos_dir = "/content/term-extraction-eval/processed_texts"  # Directorio con los archivos de texto
terminos_path = "terminos_limpios.txt"  # Ruta del archivo de términos

# Verificar existencia de los archivos
if not os.path.exists(terminos_path):
    print("❌ El archivo de términos no existe.")
    exit()

if not os.path.exists(documentos_dir):
    print(f"❌ La carpeta '{documentos_dir}' no existe.")
    exit()

# Cargar términos candidatos
with open(terminos_path, 'r', encoding='utf-8') as f:
    terminos = [line.strip() for line in f.readlines() if line.strip()]

# Leer los documentos como una lista
documentos = []
nombres_documentos = []

for archivo in sorted(os.listdir(documentos_dir)):  # Ordenamos para consistencia
    ruta = os.path.join(documentos_dir, archivo)
    if os.path.isfile(ruta) and archivo.endswith(".txt"):
        with open(ruta, 'r', encoding='utf-8') as f:
            documentos.append(f.read())
            nombres_documentos.append(archivo)

# Verificar que hay documentos para procesar
if not documentos:
    print("❌ No se encontraron documentos de texto en la carpeta especificada.")
    exit()

print(f"✅ Se han cargado {len(documentos)} documentos.")

# Número total de documentos en el corpus
total_documentos = len(documentos)

# 1️⃣ Paso: Calcular TTF (Total Term Frequency) en todo el corpus
vectorizer_ttf = CountVectorizer(vocabulary=terminos, ngram_range=(2, 3))
ttf_matrix = vectorizer_ttf.fit_transform(documentos)  # Matriz documento-término

# Sumar todas las ocurrencias de cada término en el corpus
ttf_scores = np.asarray(ttf_matrix.sum(axis=0)).flatten()

# 2️⃣ Paso: Calcular IDF correctamente con documentos individuales
df_t = (ttf_matrix > 0).sum(axis=0)  # Número de documentos donde aparece cada término
idf_scores = np.log((total_documentos + 1) / (df_t + 1)) + 1  # Ajuste para evitar división por cero

# 3️⃣ Paso: Aplicar la fórmula adaptada de TF-IDF (TTF-IDF)
ttf_idf_scores = ttf_scores * idf_scores.A1  # Multiplicación elemento a elemento

# 4️⃣ Calcular el porcentaje de documentos en los que aparece cada término
porcentaje_documentos = (df_t / total_documentos) * 100  # Convertir a porcentaje

# Crear un DataFrame con los términos y sus valores adaptados de TTF-IDF
df_tfidf = pd.DataFrame({
    "Término": vectorizer_ttf.get_feature_names_out(),
    "TTF": ttf_scores,
    "IDF": idf_scores.A1,
    "TTF-IDF": ttf_idf_scores,
    "% de Textos": porcentaje_documentos.A1  # Extraer valores de la matriz
})

# Ordenar por TTF-IDF
df_sorted_ttf_idf = df_tfidf.sort_values(by="TTF-IDF", ascending=False)

# Guardar resultados en archivo CSV
output_ttf_idf = "ranking_ttf_idf.csv"
df_sorted_ttf_idf.to_csv(output_ttf_idf, sep=",", index=False)

print(f"✅ Se ha generado el ranking de términos basado en TTF-IDF:")
print(f"📂 Archivo guardado en: {output_ttf_idf}")
