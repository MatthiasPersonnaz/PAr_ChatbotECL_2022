# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:46:25 2022

@author: matth
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
# from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import normalize


import gensim


path = './word2vec_docs_scol_traités/'
vec = np.load(path+'word2vec-phrases-vecteurs.npy')[:,:8,:] # x premières phrases; x premiers mots de chaque phrase; dimensions
dim = np.shape(vec)




print(np.shape(vec))
nb_phrases = dim[0]    # samples
nb_mots_max = dim[1]   # timesteps
embedding_dim = dim[2] # features



latent_dim = 128 # dimension du vecteur de pensée


d = nb_mots_max * embedding_dim




autoencoder = tf.keras.Sequential()

autoencoder.add(layers.LSTM(128,return_sequences = True))
autoencoder.add(layers.LSTM(128,return_sequences = False))
autoencoder.add(layers.RepeatVector(nb_mots_max))
autoencoder.add(layers.LSTM(128,return_sequences = True))
autoencoder.add(layers.LSTM(embedding_dim,return_sequences = True))
autoencoder.add(layers.Dense(embedding_dim))





autoencoder.compile(optimizer='Adam', loss=losses.CosineSimilarity(), metrics =["mse", "accuracy", "cosine_similarity"])




history = autoencoder.fit(vec, vec,
                epochs=1000,
                batch_size=16, # il semble que ça marche mieux quand batch augmente (et ça va plus vite tout bénéf)
                shuffle=True,
                validation_data=(vec, vec),
                validation_split=0.1,
                verbose=1)

autoencoder.summary()


plot_model(autoencoder, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')



plt.plot(history.history["cosine_similarity"])
plt.grid()
plt.xlabel("epoch")
plt.ylabel("cosine similarity")
plt.title("métrique VS itération")



