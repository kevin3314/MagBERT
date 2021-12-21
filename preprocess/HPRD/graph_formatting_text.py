#!/usr/bin/env python
# coding: utf-8

# Program extracting first column 
import xlrd
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import PCA

loc = ("HPRD50_multimodal_dataset.xlsx")

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0) 
sheet.cell_value(0, 0) 
PPI = []

for i in range(sheet.nrows): 
    temp = []
    for j in range(sheet.ncols):
        temp.append(sheet.cell_value(i, j))
    PPI.append(temp)


## Vocabulary
corpus = []
labels = []
proteins = []
for line in PPI:
    proteins.append(line[4:6])
    corpus.append(remove_stopwords(line[1].lower()))
    labels.append(line[-1])


vocab = []
for doc in corpus:
    vocab += doc.split()

vocab = list(set(vocab))
values = np.array(vocab)


## Encode the Sentences / Documents
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print("integer_encoded.size:", integer_encoded.size)


def encode_sentence(sentence):
    indices = label_encoder.transform(sentence.split())
    vector = np.zeros(integer_encoded.size, dtype = 'int')
    for index in indices:
        vector[index] = 1
    return(vector)


enc_corpus = []
for doc, label in zip(corpus, labels):
    enc_corpus.append(encode_sentence(doc))

pca = PCA(n_components=1000, svd_solver='arpack')
red_dim_corp = pca.fit_transform(enc_corpus)


# # Node formatting
index = 283
nodes = []
for doc, label in zip(red_dim_corp, labels):
    temp = ''
    temp += str(index) + '\t'
    temp += '\t'.join([str(n) for n in doc])
    temp += '\t'+ str(float(label)) + '\n'
    index += 1
    nodes.append(temp)


with open('./data/node_text', 'w') as f:
    f.writelines(nodes)


# # Edge Formatting
protein_corpus = []
for doc_proteins in proteins:
    for protein in doc_proteins:
        protein_corpus.append(protein)
        
protein_corpus = list(set(protein_corpus))


protein_encoder = LabelEncoder()
protein_encodings = protein_encoder.fit_transform(protein_corpus)


doc_protein_encodings = []
for protein in proteins:  
    doc_protein_encodings.append(protein_encoder.transform(protein))


doc_protein_encodings


edges = []
for i in tqdm(range(len(doc_protein_encodings))):
    for j, compare_to in enumerate(doc_protein_encodings):
        for compare in doc_protein_encodings[i]:
            if i != j and compare in compare_to:
                edges.append(str(i + 283) + "\t" + str(j + 283) + "\n")
                break

with open('./data/link_text', 'w') as f:
    f.writelines(edges)