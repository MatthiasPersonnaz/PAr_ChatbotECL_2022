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
print('texte segmenté par NLTK')

nb_max_phrases = 20000
embedding_dim = 32
latent_dim = 32 # pour plus tard dans le code


phrases = [[tok.text for tok in nlp(phrase.lower())] for phrase in texte_segmente[:nb_max_phrases]]



print('phrases tokénisées par spacy')

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
        
        
        
        
skipgram = gensim.models.Word2Vec(sentences=phrases, window=5, min_count=1, sg=1, vector_size=embedding_dim, workers=4, compute_loss=True, callbacks=[callback()],epochs=10)

print('skipgram achevé')

long_max_phrases = max([len(phrase) for phrase in phrases])



tenseur_enc_in  = np.zeros((nb_max_phrases, long_max_phrases, embedding_dim), dtype='f')

    
for p in range(len(phrases)):
    for m in range(len(phrases[p])):
        tenseur_enc_in[p][m][:] = skipgram.wv[phrases[p][m]]



print('remplissage tenseur effecuté')



from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model




enc_gru = layers.GRU(latent_dim, return_state=True)
dec_gru = tf.keras.layers.GRU(embedding_dim, return_sequences=True, return_state=True)
dec_dense = tf.keras.layers.Dense(embedding_dim, activation="tanh")



# Define an input sequence and process it.
enc_in = tf.keras.Input(shape=(None, embedding_dim))

# We discard `encoder_outputs` and only keep the states.
_, state_h = enc_gru(enc_in)
enc_states = [state_h]


# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
dec_in = tf.keras.Input(shape=(None, embedding_dim))



dec_out, _ = dec_gru(dec_in, initial_state=enc_states)

final_out = dec_dense(dec_out)

autoencoder = Model([enc_in, dec_in], final_out)



autoencoder.compile(optimizer='Adam', loss=losses.CosineSimilarity(), metrics =["mse", "accuracy", "cosine_similarity"]) # losses.MeanSquaredError() losses.CosineSimilarity()




history = autoencoder.fit([tenseur_enc_in, tenseur_enc_in],tenseur_enc_in,
                epochs=100,
                batch_size=64,
                shuffle=True,
                validation_data=([tenseur_enc_in, tenseur_enc_in],tenseur_enc_in),
                validation_split=0.1,
                verbose=1)


# autoencoder.build(input_shape=(None,None,embedding_dim))
autoencoder.summary()


plot_model(autoencoder,
           show_shapes=True,
           to_file='reconstruct_lstm_autoencoder.png',
           show_layer_names=True,
           expand_nested=False)


    
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


def autoencodeur(phrase,contexte):
    vec_wv,long_wv = phrase2vec(phrase)
    vec_cx,long_cx = phrase2vec(contexte)
    vec_decode = autoencoder.predict([vec_wv,vec_cx])[:long_cx]
    phrase_decodee = ''
    for m in range(long_cx):
        phrase_decodee += skipgram.wv.similar_by_vector(vec_decode[0][m])[0][0] + ' '
    return phrase_decodee

