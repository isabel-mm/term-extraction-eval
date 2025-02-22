import os
import re
import numpy as np
import pandas as pd
from collections import Counter

# Definir nombres de archivo
archivo_terminos = "terminos_limpios.txt"
archivo_corpus = "corpus_completo_procesado.txt"  # Cambia el nombre si tu archivo tiene otro nombre

# Cargar términos candidatos
def cargar_terminos(archivo_terminos):
    if not os.path.exists(archivo_terminos):
        print(f"Error: No se encontró {archivo_terminos}")
        return []
    
    with open(archivo_terminos, "r", encoding="utf-8") as f:
        terminos = [line.strip().lower() for line in f.readlines()]
    return terminos

# Cargar texto del corpus
def cargar_corpus(archivo_corpus):
    if not os.path.exists(archivo_corpus):
        print(f"Error: No se encontró {archivo_corpus}")
        return ""
    
    with open(archivo_corpus, "r", encoding="utf-8") as f:
        return f.read().lower()

# Contar ocurrencias de los términos en el corpus
def contar_ocurrencias(terminos, texto):
    conteo = Counter()
    for termino in terminos:
        conteo[termino] = len(re.findall(r'\b' + re.escape(termino) + r'\b', texto))
    return conteo

# Calcular C-Value
def calcular_cvalue(terminos, conteo_ocurrencias):
    cvalue_scores = {}
    terminos_set = set(terminos)

    for termino in terminos:
        f_t = conteo_ocurrencias[termino]  # Frecuencia del término
        t_length = len(termino.split())  # Longitud del término en palabras
        
        # Encontrar términos que contienen al actual
        sub_terminos = [t for t in terminos_set if t != termino and termino in t]
        P_T = len(sub_terminos)  # Número de términos que contienen este término
        
        # Frecuencia de los términos que contienen al actual
        F_T = sum(conteo_ocurrencias[t] for t in sub_terminos)
        
        if P_T > 0:
            cvalue = np.log2(t_length) * (f_t - (F_T / P_T))
        else:
            cvalue = np.log2(t_length) * f_t
        
        cvalue_scores[termino] = cvalue

    return cvalue_scores

if __name__ == "__main__":
    terminos = cargar_terminos(archivo_terminos)
    texto_corpus = cargar_corpus(archivo_corpus)
    
    if not terminos or not texto_corpus:
        print("Error: Asegúrate de que ambos archivos existen y tienen contenido.")
    else:
        conteo_ocurrencias = contar_ocurrencias(terminos, texto_corpus)
        cvalue_scores = calcular_cvalue(terminos, conteo_ocurrencias)
        
        # Guardar resultados en CSV
        df_cvalue = pd.DataFrame(cvalue_scores.items(), columns=["Término", "C-Value"])
        df_cvalue = df_cvalue.sort_values(by="C-Value", ascending=False)
        df_cvalue.to_csv("cvalue_resultados.csv", encoding="utf-8", index=False)
        
        print("Cálculo de C-Value completado. Resultados guardados en 'cvalue_resultados.csv'.")
