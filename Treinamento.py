# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 04:25:37 2020

@author: Luique
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:02:47 2020

@author: Luique
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense,Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
import time
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score





def build_model(params :list, image_x, image_y):
    num_of_classes = 10
    model = Sequential()
    model.add(Conv2D(params[0], (5, 5), input_shape=(image_x,image_y,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    model.add(Conv2D(params[1], (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(params[2], activation='relu'))
    model.add(Dropout(0.6))
    
    model.add(Dense(params[3], activation='relu'))
    model.add(Dropout(0.6))
     
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




def keras_model(image_x, image_y):
    num_of_classes = 10
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x,image_y,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.6))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
     
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "QuickDraw.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels



def prepress_labels(labels):
    labels = np_utils.to_categorical(labels)
    return labels

def reverse_one_hot(y_pred):
    n_pred = []
    for pred in y_pred:
        n_class = np.argmax(pred)
        n_pred.append(n_class)
    return n_pred

def calcular_metricas(y_pred, y_true):
    # Reverte One Hot Encoding
    y_true = reverse_one_hot(y_true)

    # Calculando F1
    f1_valor = f1_score(y_true, y_pred, average='weighted')

    # Calculando Precisão
    predicao_valor = precision_score(y_true, y_pred, average='weighted')

    # Calculando Acurácia
    acurácia_valor = accuracy_score(y_true, y_pred)

    # Calculando Recall Score
    recall_valor = recall_score(y_true, y_pred, average='weighted')

    return [f1_valor, predicao_valor, acurácia_valor, recall_valor]

def salvar_metricas(valores, tempo):
    dados = ['CNN']
    for valor in valores:
        dados.append(valor)
    dados.append(tempo)
    dados = np.array(dados)
    dados = np.reshape(dados, (1,6))

    # Salvando Metricas
    data = pd.DataFrame(dados, columns=['Modelo','F1','Precisão', 'Acurácia','Recall', 'Tempo de Predição'])
    data.to_csv('metricas/valoresCnn.csv', index=False, sep=';')

def multiple_models_fit(train_x, train_y, test_x, test_y, n_train :int):
    for i in range(0, n_train):
        # Instancia novo modelo
        model, callbacks_list = keras_model(28,28)
        # Começa Treino
        start = time.time()
        modelo_treinado = model.fit(train_x, train_y, validation_data=(test_x, test_y),
              epochs=3, batch_size=64,
              callbacks=[TensorBoard(log_dir="QuickDraw")])
        elapsed = time.time() - start

        # Salva Historico
        dataframe = pd.DataFrame(modelo_treinado.history)
        dataframe.to_csv('metricas/training_data_ %d_%.2f_seconds.csv' % (i, elapsed), index=False)

def main():
    features, labels = loadFromPickle()

    features, labels = shuffle(features, labels)
    labels=prepress_labels(labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
    model, callbacks_list = keras_model(28,28)

    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=5, batch_size=64,
              callbacks=[TensorBoard(log_dir="QuickDraw")])
    model.save('modulos/QuickDraw.h5')
    
    # grid_search(train_x, train_y, test_x, test_y) # Faz Grid Search
    # multiple_models_fit(train_x, train_y, test_x, test_y, 30) # Treina Multiplos Modelos
    
    # Calcula Metricas
    start = time.time()
    y_pred = model.predict_classes(test_x)
    end = time.time()
    elapsed = end - start
    print('Tempo de Classificação :%.2f segundos' % (elapsed))

    # Salvando Metricas
    scores = calcular_metricas(y_pred, test_y)
    salvar_metricas(scores, elapsed)

def grid_search(X, Y, x_test, y_test):

    layer_1 = [64,128]
    layer_2 = [64,128]
    layer_3 = [256,512]
    layer_4 = [64,128]
    for a in layer_1:
        for b in layer_2:
            for c in layer_3:
                for d in layer_4:
                    model = build_model([a,b,c,d],28,28)
                    model.fit(x=X, y=Y, batch_size=64, epochs=3, verbose=1, validation_data=(x_test, y_test))    
        
   

main()
