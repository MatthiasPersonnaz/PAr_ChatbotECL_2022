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
vec = np.load(path+'word2vec-phrases-vecteurs.npy')[:,:16,:] # x premières phrases; x premiers mots de chaque phrase; dimensions
dim = np.shape(vec)




print(np.shape(vec))
nb_phrases = dim[0]    # samples     (nb de phrases)
nb_mots_max = dim[1]   # timesteps   (nb max de mots par phrase)
embedding_dim = dim[2] # features    (dim du word2vec)



latent_dim = 256 # dimension du vecteur de pensée


d = nb_mots_max * embedding_dim




autoencoder = tf.keras.Sequential()



autoencoder.add(tf.keras.Input(shape=(None, embedding_dim)))
autoencoder.add(layers.LSTM(latent_dim,return_sequences = True,return_state=False))
autoencoder.add(layers.LSTM(latent_dim,return_sequences = True,return_state=False))
autoencoder.add(layers.LSTM(embedding_dim,return_sequences = True))



autoencoder.compile(optimizer='Adam', loss=losses.CosineSimilarity(), metrics =["mse", "accuracy", "cosine_similarity"]) # losses.MeanSquaredError() losses.CosineSimilarity()




history = autoencoder.fit(vec, vec,
                epochs=400,
                batch_size=16, # il semble que ça marche mieux quand batch augmente (et ça va plus vite tout bénéf)
                shuffle=True,
                validation_data=(vec, vec),
                validation_split=0.2,
                verbose=1)

autoencoder.summary()


plot_model(autoencoder, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')





plt.figure(1)
plt.plot(history.history['val_cosine_similarity'])
plt.plot(history.history['val_cosine_similarity'])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend(['Entraînement', 'Test'], loc='upper left')


plt.figure(2)
plt.plot(np.log10(history.history['mse']))
plt.plot(np.log10(history.history['val_mse']))
plt.grid()
plt.xlabel('epoch')
plt.ylabel('log mse')
plt.title('Mean Square Error')
plt.legend(['Entraînement', 'Test'], loc='upper left')



skipgram = gensim.models.Word2Vec.load(path+"Skip-Gram.model")


def phrase2vec(phrase):
    mots = phrase.split(' ')
    t = np.zeros((1,len(mots),embedding_dim),dtype='f') # floating point
    for m in range(len(mots)):
        t[0][m][:] = skipgram.wv[mots[m]]
    return t,len(mots)


def autoencodeur(phrase):
    vec_wv,longueur = phrase2vec(phrase)
    vec_decode = autoencoder.predict(vec_wv)[:longueur]
    phrase_decodee = ''
    for m in range(longueur):
        phrase_decodee += skipgram.wv.similar_by_vector(vec_decode[0][m])[0][0] + ' '
    return phrase_decodee #, vec_decode
    
   
phrases = ["vous pouvoir demander un mobilité à le commission de école",
           "le école centrale de lyon être un école de ingénieur",
           "vous devoir donc le contacter en amont pour le informer",
           "le directeur pouvoir accorder un césure sans informer",
           "le jury renvoyer le directeur car il ne avoir pas valider tout son semestre"]
 
for phrase in phrases:
    print(phrase+'\n'+autoencodeur(phrase)+'\n\n')



