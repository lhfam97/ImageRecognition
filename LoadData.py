# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:05:51 2020

@author: Luique
"""

import numpy as np
import os
import pickle

files = os.listdir("E:\source\AtividadeFinalPython\QuickDraw\\normalizedData")
x = []
x_load = []
y = []
y_load = []



def carregar_dados():
    count = 0
    for file in files:
        #file = "E:\source\AtividadeFinalPython\QuickDraw\normalizedData\\" + file
        x = np.load("normalizedData\\"+file)
        x = x.astype('float32') / 255.
        # Somente pegaremos as 10000 primeiras imagens
        x = x[0:10000, :]
        x_load.append(x)
        y = [count for _ in range(10000)]
        count += 1
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        y_load.append(y)

    return x_load, y_load


features, labels = carregar_dados()
features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')
features=features.reshape(features.shape[0]*features.shape[1],features.shape[2])
labels=labels.reshape(labels.shape[0]*labels.shape[1],labels.shape[2])

# Armazenando features e labels no Pickle
with open("features", "wb") as f:
    pickle.dump(features, f, protocol=4)
with open("labels", "wb") as f:
    pickle.dump(labels, f, protocol=4)
