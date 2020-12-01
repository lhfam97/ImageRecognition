# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:05:51 2020

@author: Luique
"""


import numpy as np
import os



# Listar diretórios onde estão os dados
files = os.listdir("E:\source\AtividadeFinalPython\QuickDraw\data")


def carregar_datasets(files :list):
    print('Baixando os 10 datasets!')   
    datasets = []
    for file in files:
        dataset = np.load("data\\"+file)
        datasets.append(dataset)
    return datasets


def normalizar(datasets :list):
    print('Normalizando Base!')
    normalized_datasets = []
    for dataset in datasets:
        images = []
        for image in dataset:
            n = image.astype('float64')
            n = (n - 127.5) / 127.5
            images.append(n)
        normalized_datasets.append(images)
        print('Base Normalizada!')
    return normalized_datasets
    

# Listar diretórios onde estão os dados
files = os.listdir("E:\source\AtividadeFinalPython\QuickDraw\data")
datasets = carregar_datasets(files)


# Normaliza valor dos pixels entre -1 e 1
datasets = normalizar(datasets)

# Salvar os dados apór normalização
 
count = 0
for file in files:
    np.save("normalizedData\\" + file, datasets[count])
    count +=1

