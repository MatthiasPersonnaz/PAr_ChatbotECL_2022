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

nb_max_sentences = 10
embedding_dim = 8
latent_dim = 4 # pour plus tard dans le code


sentences = [[tok.text for tok in nlp(sent.lower())] for sent in texte_segmente[:nb_max_sentences]]



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
        
        
        
        
skipgram = gensim.models.Word2Vec(sentences=sentences, window=5, min_count=1, sg=1, vector_size=embedding_dim, workers=4, compute_loss=True, callbacks=[callback()],epochs=3)

print('skipgram achevé')

sentence_max_size = max([len(sent) for sent in sentences])



tenseur_enc_in  = np.zeros((nb_max_sentences, sentence_max_size, embedding_dim), dtype=float)

    
for p in range(len(sentences)):
    for m in range(len(sentences[p])):
        tenseur_enc_in[p][m][:] = skipgram.wv[sentences[p][m]]



print('remplissage tenseur effecuté')



from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model



# je suis le modèle de https://stackoverflow.com/questions/68280503/how-to-add-the-attention-layer-in-the-tensorflow-gru-model

## ENCODER
enc_in  = tf.keras.Input(shape=(None, embedding_dim))

enc_gru = layers.GRU(latent_dim, return_sequences=True, return_state=True)

enc_out, enc_state = enc_gru(enc_in)

encoder = Model(enc_in,[enc_out,enc_state], name='encoder')



## DECODER
dec_in        = tf.keras.Input(shape=(None, latent_dim)) # comme enc_in
dec_inp_state = tf.keras.Input(shape=(latent_dim))
dec_enc_out   = tf.keras.Input(shape=(None, latent_dim))


dec_gru = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)


dec_out, dec_state = dec_gru(dec_in, initial_state = dec_inp_state) 
attention = tf.keras.layers.Attention()([dec_enc_out, dec_out])
concat = tf.keras.layers.Concatenate()([attention, dec_out])
final_output = tf.keras.layers.Dense(embedding_dim)(concat)


decoder = Model([dec_in, dec_inp_state, dec_enc_out], final_output)

# COMPLETE AUTOENCODER
inp_e = tf.keras.Input(shape=(None, embedding_dim))
o_e, h_e = encoder(inp_e)
inp_d = tf.keras.Input(shape=(None, latent_dim))
out = decoder([inp_d, h_e, o_e])

autoencoder = Model([inp_e, inp_d], out)
autoencoder.compile(optimizer='Adam', loss=losses.CosineSimilarity(), metrics =["mse", "accuracy", "cosine_similarity"])


autoencoder.compile() # losses.MeanSquaredError() losses.CosineSimilarity() CategoricalCrossEntropy()


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


def sent2vec(sent):
    sent = sent.lower()
    mots = sent.split(' ')
    t = np.zeros((1,len(mots),embedding_dim),dtype='f') # floating point
    for m in range(len(mots)):
        t[0][m][:] = skipgram.wv[mots[m]]
    return t,len(mots)


def autoencodeur(sent):
    vec_wv,longueur = sent2vec(sent)
    vec_decode = autoencoder.predict([vec_wv,vec_wv[:][-1]])[:longueur]
    sent_decodee = ''
    for m in range(longueur):
        sent_decodee += skipgram.wv.similar_by_vector(vec_decode[0][m])[0][0] + ' '
    return sent_decodee

