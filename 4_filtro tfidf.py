import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Definir rutas de archivos
corpus_path = "/content/term-extraction-eval/corpus_completo_procesado.txt"
terminos_path = "terminos_limpios.txt"  # Ruta del archivo de términos

# Verificar existencia de los archivos
if not os.path.exists(terminos_path):
    print("❌ El archivo de términos no existe.")
    exit()

if not os.path.exists(corpus_path):
    print("❌ El archivo del corpus no existe.")
    exit()

# Cargar términos candidatos, eliminando líneas vacías
with open(terminos_path, 'r', encoding='utf-8') as f:
    terminos = [line.strip() for line in f.readlines() if line.strip()]

# Cargar el corpus completo en una única variable
with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus_text = f.read()

print("✅ Corpus cargado exitosamente.")

# Definir número total de documentos en el corpus
total_documentos = 30  # 🔹 Ajusta este valor si tienes el número exacto

# 1️⃣ Paso: Calcular TTF (Total Term Frequency) en todo el corpus
vectorizer_ttf = CountVectorizer(vocabulary=terminos, ngram_range=(2, 3))
ttf_matrix = vectorizer_ttf.fit_transform([corpus_text])  # Matriz con una sola fila (todo el corpus)

# Sumar todas las ocurrencias de cada término en el corpus
ttf_scores = np.asarray(ttf_matrix.sum(axis=0)).flatten()

# 2️⃣ Paso: Calcular IDF con la fórmula adaptada
df_t = (ttf_matrix > 0).sum(axis=0)  # Número de documentos donde aparece cada término
idf_scores = np.log((total_documentos + 1) / (df_t + 1)) + 1  # Ajuste para evitar división por cero

# 3️⃣ Paso: Aplicar la fórmula adaptada de TF-IDF (TTF-IDF)
ttf_idf_scores = ttf_scores * idf_scores.A1  # Multiplicación elemento a elemento

# Crear un DataFrame con los términos y sus valores adaptados de TF-IDF
df_tfidf = pd.DataFrame({
    "Término": vectorizer_ttf.get_feature_names_out(),
    "TTF": ttf_scores,
    "IDF": idf_scores.A1,
    "TTF-IDF": ttf_idf_scores
})

# Ordenar por TTF-IDF
df_sorted_ttf_idf = df_tfidf.sort_values(by="TTF-IDF", ascending=False)

# Guardar resultados en archivo CSV
output_ttf_idf = "ranking_ttf_idf.csv"
df_sorted_ttf_idf.to_csv(output_ttf_idf, sep=",", index=False)

print(f"✅ Se ha generado el ranking de términos basado en TTF-IDF:")
print(f"📂 Archivo guardado en: {output_ttf_idf}")
