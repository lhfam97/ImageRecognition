# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:52:40 2020

@author: Luique
"""

import cv2
import numpy as np
from keras.models import load_model
model = load_model('modulos/QuickDraw.h5')

def process_image(img):
    # Converte para Grayscale
    n = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensiona imagem
    n = cv2.resize(n, (640, 480))

    # Rect Value
    frame_w = 640
    frame_h = 480
    box_size = 200

    fw_size = (frame_w//2)
    fh_size = (frame_h//2)
    b_size = box_size//2

    start_x = fw_size - b_size
    start_y = fh_size - b_size
    end_x = fw_size + b_size
    end_y = fh_size + b_size

    #n = n[start_y:end_y, start_x:end_x]

    # Suavizando Imagem para tirar ruidos
    n = cv2.GaussianBlur(n, (3, 3), 0)

    # Binariza Imagem usando um Treshold Adaptavel
    n = cv2.adaptiveThreshold(n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)

    # Suavizando Imagem para aumentar espessura do traço
    n = cv2.GaussianBlur(n, (11, 11), 0)

    # Binariza Imagem
    n = cv2.adaptiveThreshold(n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)

    # Inverte cor
    n = ~n

    return n



cap = cv2.VideoCapture('http://192.168.86.53:4747/video')

def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = (img - 127.5)/127.5
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

while True:
    ret, frame = cap.read()
    img = process_image(frame)
    cv2.imshow('window', img)
    cv2.waitKey(15)
    pred_probab, pred_class = keras_predict(model, img)
    dictionary = {0:"apple", 1:"banana", 2: "book", 3: "car", 4: "fork", 5: "hurricane", 6: "mug", 7: "pizza", 8: "star"
                  ,9:"t-shirt"}
    print(dictionary[pred_class], pred_probab)
    print(pred_class, pred_probab)
cap.release()
cv2.destroyAllWindows()