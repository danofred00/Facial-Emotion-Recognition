# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:26:47 2024
@author: MBAKOP AMBROISE CARTEL JONAS
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple

import cv2


# pretrained model save path
MODEL_PATH = 'packages/backend/model/data/'

# Fonction pour charger les données depuis le fichier CSV
def load_data(csv_file) -> Tuple[np.array, np.array, pd.DataFrame]:
    # Lire le fichier CSV
    data = pd.read_csv(csv_file)
    
    # Extraire les séquences de pixels
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []

    # Convertir les séquences de pixels en matrices 48x48
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)       # Ajouter une dimension pour les canaux
    emotions = pd.get_dummies(data['emotion']).values      # Convertir les émotions en vecteurs one-hot
    return faces, emotions, data

# Charger les données à partir du fichier CSV
dataset = 'fer2013.csv'
print(f"[+] Loading data from dataset {dataset} ... ", end=' ')
faces, emotions, data = load_data(dataset)
print('done')

# Diviser les données en ensembles d'entraînement et de test 
# Les indices où la colonne 'Usage' est 'Training' et 'PublicTest'

print("[+] Spliting datas into Training and PublicTest")

train_idx = data['Usage'] == 'Training'
test_idx = data['Usage'] == 'PublicTest'

train_images = faces[train_idx]
train_labels = emotions[train_idx]
test_images = faces[test_idx]
test_labels = emotions[test_idx]

# Normaliser les données
# Diviser les valeurs des pixels par 255 pour les ramener entre 0 et 1
train_images = train_images / 255.0
test_images = test_images / 255.0


# Prétraitement des données avec augmentation
print("[+] Define the image data generator")
datagen = ImageDataGenerator(
    rotation_range=20,        # Rotation des images jusqu'à 20 degrés
    width_shift_range=0.2,    # Déplacement horizontal des images jusqu'à 20%
    height_shift_range=0.2,   # Déplacement vertical des images jusqu'à 20%
    zoom_range=0.2,           # Zoom sur les images jusqu'à 20%
    shear_range=0.15,         # Cisaillement des images jusqu'à 15%
    horizontal_flip=True,     # Flipping horizontal des images
    fill_mode='nearest'       # Mode de remplissage pour les pixels manquants
)
datagen.fit(train_images)     # Appliquer l'augmentation sur les données d'entraînement


# Définir le modèle CNN amélioré
print("[+] Creating a tensorflow sequential model ... ", end='')
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),  # Première couche convolutionnelle avec 32 filtres
    layers.MaxPooling2D((2, 2)),                                            # Première couche de pooling (réduction de dimension)
    layers.Conv2D(64, (3, 3), activation='relu'),                           # Deuxième couche convolutionnelle avec 64 filtres
    layers.MaxPooling2D((2, 2)),                                            # Deuxième couche de pooling
    layers.Conv2D(128, (3, 3), activation='relu'),                          # Troisième couche convolutionnelle avec 128 filtres
    layers.MaxPooling2D((2, 2)),                                            # Troisième couche de pooling
    layers.Conv2D(256, (3, 3), activation='relu'),                          # Quatrième couche convolutionnelle avec 256 filtres
    layers.Flatten(),                                                       # Aplatir les données pour la couche fully connected
    layers.Dense(256, activation='relu'),                                   # Première couche fully connected avec 256 neurones
    layers.Dropout(0.5),                                                    # Couche de dropout pour éviter le surapprentissage
    layers.Dense(128, activation='relu'),                                   # Deuxième couche fully connected avec 128 neurones
    layers.Dropout(0.5),                                                    # Deuxième couche de dropout
    layers.Dense(7, activation='softmax')                                   # Couche de sortie avec activation softmax pour les 7 émotions
])
print("Done")


# Compiler le modèle avec un taux d'apprentissage ajusté
print("[+] Compiling the model")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Optimiseur Adam avec un taux d'apprentissage spécifique
              loss='categorical_crossentropy',                           # Fonction de perte pour les labels one-hot
              metrics=['accuracy'])                                      # Métrique d'évaluation


# Entraîner le modèle
print("[+] Training the model ... ", end='')
history = model.fit(datagen.flow(train_images, train_labels, batch_size=64),  # Utiliser l'augmentation des données pour l'entraînement
                    epochs=100,                                                 # Nombre d'époques (cycle d'entrainement complets)
                    validation_data=(test_images, test_labels))               # Données de validation pour évaluer la performance
print("Done")


# Évaluer le modèle
print("[+] Evaluating the model ...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)  # Évaluer le modèle sur les données de test
print("Done")
print('[+] Model accuracy : ', test_acc)          # Afficher la précision du modèle


# Afficher les courbes de perte et de précision
plt.plot(history.history['accuracy'], label='Précision Entraînement')  # Courbe de précision d'entraînement
plt.plot(history.history['val_accuracy'], label='Précision Validation')  # Courbe de précision de validation
plt.xlabel('Époque')  # Label pour l'axe x
plt.ylabel('Précision')  # Label pour l'axe y
plt.legend(loc='lower right')  # Légende positionnée en bas à droite
plt.show()  # Afficher la courbe

plt.plot(history.history['loss'], label='Perte Entraînement')  # Courbe de perte d'entraînement
plt.plot(history.history['val_loss'], label='Perte Validation')  # Courbe de perte de validation
plt.xlabel('Époque')  # Label pour l'axe x
plt.ylabel('Perte')  # Label pour l'axe y
plt.legend(loc='upper right')  # Légende positionnée en haut à droite
plt.show()  # Afficher la courbe


# Sauvegarder le modèle
path = os.path.join(MODEL_PATH, 'facial_emotion_recognition_model.h5')
model.save(path)  # Sauvegarder le modèle entraîné dans un fichier .h5
print(f"[+] Model saved at : {path}")

