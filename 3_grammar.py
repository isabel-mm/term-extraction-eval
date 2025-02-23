import os
import re
import nltk
from collections import Counter

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

GRAMMAR = r"""
    NP: 
        {<JJ>*<NN.*><NN.*>+}    
        {<JJ>*<NN.*><IN><NN.*>+} 
        {<NN.*><NN.*>} 
        {<NN.*><IN><NN.*>} 
"""

CORPUS_FILE = "corpus_completo_procesado.txt"
OUTPUT_FILE = "terminos_extraidos_filtrados.txt"

if not os.path.exists(CORPUS_FILE):
    print(f"❌ El archivo {CORPUS_FILE} no se encuentra en el directorio.")
    exit()

print("📂 Cargando el corpus...")
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    text = f.read().lower()

print("🔍 Tokenizando y etiquetando el texto...")
sentences = nltk.sent_tokenize(text)
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
tagged_sentences = nltk.pos_tag_sents(tokenized_sentences)

print("🌱 Extrayendo candidatos a términos...")
chunk_parser = nltk.RegexpParser(GRAMMAR)
term_candidates = set()

for sent in tagged_sentences:
    tree = chunk_parser.parse(sent)
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':  
            term = " ".join(word for word, tag in subtree.leaves())
            if len(term.split()) > 1:  # Filtra términos de una sola palabra
                term_candidates.add(term)

print(f"✅ Se han extraído {len(term_candidates)} términos únicos de más de una palabra.")

print("📊 Contando frecuencia de términos en el corpus...")
term_freq = Counter(re.findall(r'\b(?:' + '|'.join(map(re.escape, term_candidates)) + r')\b', text))

filtered_terms = {term: freq for term, freq in term_freq.items() if freq > 2}
print(f"✅ Se han filtrado {len(filtered_terms)} términos con frecuencia > 2.")

print(f"💾 Guardando términos en {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for term, freq in sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{term}\t{freq}\n")

print(f"✅ Proceso finalizado: {len(filtered_terms)} términos guardados en {OUTPUT_FILE}.")
