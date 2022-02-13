# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 17:47:40 2022

@author: matth
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gensim
import nltk
import spacy
print('bibliothèques importées')

with open("./Clement's Lab/PhrasesFR.txt", 'r', encoding='utf-8') as f:
    texte = f.read()

print('texte importé')

texte_segmente = nltk.tokenize.sent_tokenize(texte, language='french')
nlp = spacy.load('fr_core_news_sm')

nb_max_phrases = 20000
embedding_dim = 32
latent_dim = 32 # pour plus tard dans le code

phrases = [[tok.text for tok in nlp(phrase)] for phrase in texte_segmente[:nb_max_phrases]]


from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1
        
        
        
        
skipgram = gensim.models.Word2Vec(sentences=phrases, window=5, min_count=1, sg=1, vector_size=embedding_dim, workers=4, compute_loss=True, callbacks=[callback()])

print('skipgram achevé')

long_max_phrases = max([len(phrase) for phrase in phrases])



tenseurPhrases = np.zeros((nb_max_phrases, long_max_phrases, embedding_dim), dtype='f')

    
for p in range(len(phrases)):
    for m in range(len(phrases[p])):
        tenseurPhrases[p][m][:] = skipgram.wv[phrases[p][m]]

print('remplissage tenseur effecuté')



from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model




class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.hidden_state = tf.keras.Input(shape=(None,latent_dim))
        self.cell_state = tf.keras.Input(shape=(None,latent_dim))
        
        # layers
        self.enc_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
        self.dec_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=False)
        self.dec_dense = layers.Dense(embedding_dim, activation='tanh')

            
    def call(self,enc_in):
        enc_out, self.hidden_state, self.cell_state = self.enc_lstm(enc_in)
        dec_out = self.dec_lstm(enc_out, initial_state=[self.hidden_state,self.cell_state])
        final_output = self.dec_dense(dec_out)
        return final_output




autoencoder = Autoencoder()

autoencoder.compile(optimizer='Adam', loss=losses.CosineSimilarity(), metrics =["mse", "accuracy", "cosine_similarity"]) # losses.MeanSquaredError() losses.CosineSimilarity()


autoencoder.build(input_shape=(None,None,embedding_dim))
autoencoder.summary()



history = autoencoder.fit(tenseurPhrases, tenseurPhrases,
                epochs=100,
                batch_size=64,
                shuffle=True,
                validation_data=(tenseurPhrases, tenseurPhrases),
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


def phrase2vec(phrase):
    phrase = phrase.lower()
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

