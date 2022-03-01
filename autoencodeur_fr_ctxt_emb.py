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

nb_max_sentences = 1000
w2v_dim = 64 # pour gensim
embedding_dim = 32 # pour le réseau
latent_dim = 32 # pour plus tard dans le code


sentences = [[tok.text for tok in nlp(sent.lower())] for sent in texte_segmente[:nb_max_sentences]]



print('sentences tokénisées par spacy')

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
        
        
        
        
skipgram = gensim.models.Word2Vec(sentences=sentences, window=5, min_count=1, sg=1, vector_size=w2v_dim, workers=4, compute_loss=True, callbacks=[callback()],epochs=10)
print('skipgram achevé')

pretrained_weights = skipgram.syn1neg
vocab_size = len(skipgram.wv)
sentence_max_size = max([len(sent) for sent in sentences])


tenseur_enc_in  = np.zeros((nb_max_sentences, sentence_max_size, w2v_dim), dtype='f')

    
for p in range(len(sentences)):
    for m in range(len(sentences[p])):
        tenseur_enc_in[p][m][:] = skipgram.wv[sentences[p][m]]



print('remplissage tenseur effecuté')

def word2idx(word):
  return skipgram.wv.key_to_index[word]
def idx2word(idx):
  return skipgram.wv.index2word[idx]


print('\nPreparing the data for LSTM...')
train_x = np.zeros([nb_max_sentences, sentence_max_size], dtype=np.int32)
train_y = np.zeros([nb_max_sentences], dtype=np.int32)
for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
    train_x[i, t] = word2idx(word)
  train_y[i] = word2idx(sentence[-1])
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)



from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


embedding_layer = layers.Embedding(input_dim = vocab_size,
                            output_dim = embedding_dim, 
                            weights=[pretrained_weights])
                            # input_length = sentence_max_size)


enc_lstm = layers.LSTM(latent_dim, return_state=True)
dec_lstm = tf.keras.layers.LSTM(embedding_dim, return_sequences=True, return_state=True)
dec_dense = tf.keras.layers.Dense(embedding_dim, activation="tanh")



# Define an input sequence and process it.
enc_in = tf.keras.Input(shape=(None, embedding_dim))

# We discard `encoder_outputs` and only keep the states.
_, state_h, state_c = enc_lstm(enc_in)
enc_states = [state_h, state_c]


# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
dec_in = tf.keras.Input(shape=(None, embedding_dim))



dec_out, _, _ = dec_lstm(dec_in, initial_state=enc_states)

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

