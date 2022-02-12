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
vec = np.load(path+'word2vec-phrases-vecteurs.npy')[:,:32,:] # x premières phrases; x premiers mots de chaque phrase; dimensions
dim = np.shape(vec)




print(np.shape(vec))
nb_phrases = dim[0]    # samples     (nb de phrases)
nb_mots_max = dim[1]   # timesteps   (nb max de mots par phrase)
embedding_dim = dim[2] # features    (dim du word2vec)



latent_dim = 256 # dimension du vecteur de pensée


d = nb_mots_max * embedding_dim

          
            

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.state_h = tf.keras.Input(shape=(None,latent_dim))
        self.state_c = tf.keras.Input(shape=(None,latent_dim))
        
        # layers
        self.encoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
        self.decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=False)
        self.decoder_dense = layers.Dense(embedding_dim, activation='tanh')

            
    def call(self,encoder_input):
        encoder_output, self.state_h, self.state_c = self.encoder_lstm(encoder_input)
        decoder_output = self.decoder_lstm(encoder_output, initial_state=[self.state_h,self.state_c])
        final_output = self.decoder_dense(decoder_output)
        return final_output
    

            

    

autoencoder = Autoencoder()

autoencoder.compile(optimizer='Adam', loss=losses.CosineSimilarity(), metrics =["mse", "accuracy", "cosine_similarity"]) # losses.MeanSquaredError() losses.CosineSimilarity()


autoencoder.build(input_shape=(None,None,embedding_dim))
autoencoder.summary()
plot_model(autoencoder,
           show_shapes=True,
           to_file='reconstruct_lstm_autoencoder.png',
           show_layer_names=True,
           expand_nested=False)



history = autoencoder.fit(vec, vec,
                epochs=400,
                batch_size=32,
                shuffle=True,
                validation_data=(vec, vec),
                validation_split=0.1,
                verbose=1)







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
plt.ylabel('mse')
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
    return phrase_decodee
    
   
phrases = ["vous pouvoir demander un mobilité à le commission de école",
           "le école centrale de lyon être un école de ingénieur",
           "vous devoir donc le contacter en amont pour le informer",
           "le directeur pouvoir accorder un césure sans informer",
           "le jury renvoyer le directeur car il ne avoir pas valider tout son semestre"]
 
for phrase in phrases:
    print(phrase+'\n'+autoencodeur(phrase)+'\n\n')


