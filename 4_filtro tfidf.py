import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Definir rutas de archivos
documentos_dir = "processed_texts"
terminos_path = "terminos_limpios.txt"  # Ruta correcta del archivo de términos

# Verificar existencia del archivo de términos
if not os.path.exists(terminos_path):
    print("❌ El archivo de términos no existe.")
    exit()

# Cargar los términos candidatos, eliminando líneas vacías
with open(terminos_path, 'r', encoding='utf-8') as f:
    terminos = [line.strip() for line in f.readlines() if line.strip()]

# Identificar todos los archivos de texto en el directorio especificado
documentos = []
nombres_documentos = []

if not os.path.exists(documentos_dir):
    print(f"❌ La carpeta '{documentos_dir}' no existe.")
    exit()

for archivo in sorted(os.listdir(documentos_dir)):  # Ordenamos para consistencia
    ruta = os.path.join(documentos_dir, archivo)
    if os.path.isfile(ruta) and archivo.endswith(".txt"):
        with open(ruta, 'r', encoding='utf-8') as f:
            documentos.append(f.read())
            nombres_documentos.append(archivo)  # Guardamos el nombre del documento

# Verificar que hay documentos para procesar
if not documentos:
    print("❌ No se encontraron documentos de texto en la carpeta especificada.")
    exit()

print(f"✅ Se han cargado {len(documentos)} documentos.")

# Crear el vectorizador TF-IDF con el vocabulario restringido a los términos extraídos
vectorizer = TfidfVectorizer(vocabulary=terminos, ngram_range=(2, 3))  # Trabajamos solo con bigramas y trigramas
tfidf_matrix = vectorizer.fit_transform(documentos)

# Verificar que el vectorizador no esté vacío
if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
    print("❌ La matriz TF-IDF está vacía. Revisa que los términos coincidan con el contenido del corpus.")
    exit()

# Obtener los términos y sus puntajes TF-IDF
tfidf_scores = tfidf_matrix.toarray()

# Crear un DataFrame con los términos y sus valores TF-IDF
df_tfidf = pd.DataFrame({
    "Término": vectorizer.get_feature_names_out(),
    "TF-IDF Promedio": tfidf_scores.mean(axis=0),
    "TF-IDF Máximo": tfidf_scores.max(axis=0)
})

# Guardar dos rankings: por TF-IDF Promedio y TF-IDF Máximo
df_sorted_promedio = df_tfidf.sort_values(by="TF-IDF Promedio", ascending=False)
df_sorted_maximo = df_tfidf.sort_values(by="TF-IDF Máximo", ascending=False)

# Crear la carpeta de salida si no existe
os.makedirs(documentos_dir, exist_ok=True)

# Guardar resultados en archivos CSV
output_promedio = os.path.join(documentos_dir, "ranking_tfidf_promedio.csv")
output_maximo = os.path.join(documentos_dir, "ranking_tfidf_maximo.csv")

df_sorted_promedio.to_csv(output_promedio, sep=",", index=False)
df_sorted_maximo.to_csv(output_maximo, sep=",", index=False)

print(f"✅ Se han generado los rankings de términos basado en TF-IDF:")
print(f"📂 Ranking por TF-IDF Promedio guardado en: {output_promedio}")
print(f"📂 Ranking por TF-IDF Máximo guardado en: {output_maximo}")
