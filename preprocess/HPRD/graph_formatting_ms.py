#!/usr/bin/env python
# coding: utf-8

import ast
import xlrd
from pypdb import *
import Bio
from Bio.PDB import PDBList
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

loc = ("HPRD50_multimodal_dataset.xlsx")

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(1) 
sheet.cell_value(0, 0) 
PPI = []

for i in range(sheet.nrows): 
    PPI.append([sheet.cell_value(i, 0), sheet.cell_value(i, 1), sheet.cell_value(i, 2)])

unique_PPI = [list(x) for x in set(tuple(x) for x in PPI)]

# Extract PDB Files using API
pdbl = PDBList()
protein_gene = {}
def gene_to_pdb(pdb_id):
    found_pdbs = Query(pdb_id).search()
    print (found_pdbs)
    temp= []
    for j in found_pdbs:
        temp.append(int(describe_pdb(j)['nr_residues']))
    final_pdb_id = found_pdbs[temp.index(max(temp))]
    protein_gene[pdb_id] = final_pdb_id
    print (final_pdb_id)
    pdbl.retrieve_pdb_file(final_pdb_id,pdir='PDB', file_format='pdb')

unique_protein = []
for proteins in unique_PPI:
    unique_protein.append(proteins[0])
    unique_protein.append(proteins[1])

unique_protein = list(set(unique_protein))
print(len(unique_protein), "instances of unique proteins found from", len(unique_PPI), "documents")

with open('./data/protein_gene.txt', 'r') as file:
    contents = file.read()
    protein_gene = ast.literal_eval(contents)

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(["N", "C", "O", "H", "S", "N1+"])

def get_feature_vector(gene):
    temp = []
    with open('./PDB/pdb' + gene.lower() +'.ent', 'r') as f:
        lines = f.readlines()
        for line in lines:
            split_ = line.split()
            if split_[0] == 'ATOM':
                temp.extend([float(split_[6]), float(split_[7]), float(split_[8])])  
                temp.extend(label_encoder.transform([split_[-1]]))#.toarray().squeeze().tolist())
    return(temp)


nodes = []
labels = []
remove = []
feature_vector = {}

for i in tqdm(range(len(unique_PPI))):
    proteins = unique_PPI[i]
    try:
        temp = []
        gene_1 = protein_gene[proteins[0]]
        gene_2 = protein_gene[proteins[1]]
        if proteins[0] not in feature_vector:
            feature_vector[gene_1] = get_feature_vector(gene_1)
#         length.append(len(feature_vector[gene_1]))
        temp.extend(feature_vector[gene_1])
        temp.extend([0] * (431520 - len(temp)))
        if proteins[1] not in feature_vector:
            feature_vector[gene_2] = get_feature_vector(gene_2)
#         length.append(len(feature_vector[gene_2]))
        temp.extend(feature_vector[gene_2])
        temp.extend([0] * (863040 - len(temp)))
        nodes.append(temp)
        labels.append(float(proteins[2]))
    
    except:
        remove.append(proteins)
#         continue

unique_PPI = [x for x in unique_PPI if x not in remove]

with open('./data/unique_ppi.txt', 'w') as filehandle:
    for listitem in unique_PPI:
        filehandle.write('%s\n' % listitem)

with open("./data/node.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(nodes)

with open("./data/label.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(labels)


## CNN Auto Encoder
nodes = []

with open('./data/node.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    for row in csv_reader:
        if(len(row) != 0):
            nodes.append([float(val) for val in row])

labels = []

with open('./data/label.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    for row in csv_reader:
        labels.extend([float(val) for val in row])

# X_train, X_test, y_train, y_test = train_test_split(nodes, labels, test_size=0, random_state=42)
X_train = nodes
y_train = labels

def pre_process(tens):
    tens = torch.reshape(tens, (tens.shape[0], -1, 4))
    tens = torch.transpose(tens, 1, 2)
    return(tens)

X_train, y_train = map(torch.tensor, (X_train, y_train))
X_train = pre_process(X_train)


class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=5, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(431516, 1000)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

encoder = Net()

nodes_enc = []
for i in tqdm(range(len(X_train))):
    encoding = [i]
    encoding.extend(encoder.forward(X_train[i].unsqueeze(0).unsqueeze(0)).squeeze().tolist())
    encoding.append(y_train[i].numpy())
    nodes_enc.append(encoding)


with open("./data/node_enc.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(nodes_enc)


lines = []
for node in nodes_enc:
    temp = "\t".join([str(elem) for elem in node])
    lines.append(temp + "\n")

with open('./data/node_MS', 'w') as f:
    f.writelines(lines)

# define empty list
unique_PPI = []

# open file and read the content in a list
with open('./data/unique_ppi.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()

    for line in filecontents:
        temp = []
        # remove linebreak which is the last character of the string
        line = line[:-1]
        line = line.strip('[').strip(']').split(", ")
        temp = [item.replace("'", "") for item in line]
        # add item to the list
        unique_PPI.append(temp)

with open('./data/protein_gene.txt', 'r') as file:
    contents = file.read()
    protein_gene = ast.literal_eval(contents)

edges = []
for i in tqdm(range(0, len(unique_PPI))):
    for j in range(i + 1, len(unique_PPI)):
        expt_1 = unique_PPI[i]
        expt_2 = unique_PPI[j]
        genes_1 = [protein_gene[unique_PPI[i][0]], protein_gene[unique_PPI[i][1]]]
        genes_2 = [protein_gene[unique_PPI[j][0]], protein_gene[unique_PPI[j][1]]]
        if(len([x for x in genes_1 if x in genes_2])):
            edges.append(str(i) + "\t" + str(j) + "\n")

with open('./data/link_MS', 'w') as f:
    f.writelines(edges)

with open('./data/node_text', 'r') as f:
    lines_node_text = f.readlines()
    
f.close()

with open('./data/node_MS', 'r') as f:
    lines_node_ms = f.readlines()
    
f.close()

lines = lines_node_ms + lines_node_text
    
with open('../Graph-Bert/data/ppi/node', 'w') as f:
    f.writelines(lines)

f.close()

with open('./data/link_text', 'r') as f:
    lines_link_text = f.readlines()
    
f.close()

with open('./data/link_MS', 'r') as f:
    lines_link_ms = f.readlines()
    
f.close()

with open('../Graph-Bert/data/ppi/link', 'w') as f:
    f.writelines(lines_link_ms)
    f.writelines(lines_link_text)

f.close()