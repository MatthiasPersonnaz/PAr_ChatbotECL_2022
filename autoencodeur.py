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
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


import gensim


path = './word2vec_docs_scol_traités/'
vec = np.load(path+'phrases-vecteurs.npy')[:,:,:] # x premières phrases, 20 premiers mots de chaque phrase et toutes les dimensions
s = np.shape(vec)

print(np.shape(vec))
nb_phrases = s[0]
nb_mots_max = s[1]
embedding_dim = s[2]
latent_dim = 512


d = nb_mots_max*embedding_dim


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   


        self.encoder = tf.keras.Sequential([
          #layers.Flatten(),
          layers.GRU(128,activation='tanh', recurrent_activation='sigmoid',kernel_initializer='glorot_uniform',return_sequences=False),
          layers.Dense(latent_dim, activation='tanh'),
        ])
        self.decoder = tf.keras.Sequential([
          layers.Dense(d, activation='tanh'),
          layers.Reshape(s[1:])
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.CosineSimilarity(), metrics =["mse", "accuracy", "cosine_similarity"]) # "categorical_crossentropy" losses.MeanSquaredError() losses.CosineSimilarity()





history = autoencoder.fit(vec, vec,
                epochs=300,
                shuffle=True,
                validation_data=(vec, vec))

autoencoder.summary()


phrases_encodees = autoencoder.encoder(vec).numpy()
phrases_decodees = autoencoder.decoder(phrases_encodees).numpy()

plt.plot(history.history["cosine_similarity"])
plt.grid()
plt.xlabel("itération")
plt.ylabel("similarité cosinus")
plt.title("métrique VS itération")
np.linalg.norm(phrases_decodees[0][0]-vec[0][0])

skipgram = gensim.models.Word2Vec.load(path+"skipgram.model")


def phrase2vec(phrase):
    mots = phrase.split(' ')
    t = np.zeros((1,nb_mots_max,embedding_dim),dtype='f') # floating point
    for m in range(len(mots)):
        t[0][m][:] = skipgram.wv[mots[m]]
    return t


def autoencodeur(phrase):
    t = phrase2vec(phrase)
    d = autoencoder.predict(t)
    l = ''
    for m in range(nb_mots_max):
        l += skipgram.wv.similar_by_vector(d[0][m])[0][0] + ' '
    return d,l
    
   
    
phrase = "vous pouvoir demander un mobilité à le commission de école"
phrase = "le école centrale de lyon être un école de ingénieur"
phrase = "vous devoir donc le contacter en amont pour le informer"
phrase = "le directeur pouvoir accorder un césure sans informer le commission"

t = phrase2vec(phrase)
print(np.shape(t))
pred,rea = autoencodeur(phrase)

print(phrase)
print(t[0][0])
print('\n')
print(rea)
print(pred[0][0])

