# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 21:46:11 2021

@author: cleme
"""



import numpy as np
import tensorflow as tf
from tensorflow import keras
import gensim
import re

batch_size = 1   # Batch size for training.
epochs = 5  # Number of epochs to train for.
latent_dim = 300  # Latent dimensionality of the encoding space.
num_samples = 1  # Number of samples to train on.
taille_vec_mot = 50
# Path to the data txt file on disk.
data_path = 'C:/Users/matth/Documents/GitHub/par102-chatbot/documents_scolarité/données_scolarité_propres/phrases.txt'



# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()




#traitement des données

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

lines = [u.replace('\n','') for u in lines if not u=='\n']
lines = [u.replace('.', '') for u in lines]


# liste des phrases lemmatisées
# liste des suites de vecteurs

import nltk
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as stopwordsSpacy
path = 'C:/Users/matth/Documents/GitHub/par102-chatbot/documents_scolarité/données_scolarité_propres/corpus.txt'

with open(path, 'r', encoding='utf-8') as f:
    texte = f.read()
phrases = nltk.tokenize.sent_tokenize(texte, language='french')  
nlp = spacy.load('fr_core_news_sm')


phrasesTokeniseesSpacy = [nlp(s) for s in phrases]

lemmatizedSentencesSpacy = []
for p in phrasesTokeniseesSpacy:
    explodedSentence = []
    for token in p:
        if not token.is_punct:
            if token.ent_type_ != '': # déterminer si le token fait partie d'une entité
                explodedSentence.append(token.text.lower()) if token.text.lower() not in stopwordsSpacy else None
            else:
                explodedSentence.append(token.lemma_.lower()) if token.text.lower() not in stopwordsSpacy else None
    lemmatizedSentencesSpacy.append(explodedSentence)


    
    
skipgram = gensim.models.Word2Vec(sentences=lemmatizedSentencesSpacy, window=6, min_count=1, sg=1, vector_size=taille_vec_mot)

skipgram.wv.add_vector('<begin>',np.ones(taille_vec_mot))
skipgram.wv.add_vector('<end>',np.zeros(taille_vec_mot))


entrees = [[skipgram.wv[u] for u in s] for s in lemmatizedSentencesSpacy]




sorties = [[[skipgram.wv['<begin>']]+skipgram.wv[u]+[skipgram.wv['<end>']] for u in s] for s  in lemmatizedSentencesSpacy]





max_encoder_seq_length = max([len(u) for u in entrees])
max_decoder_seq_length = max([len(u) for u in sorties])

print("Number of samples:", len(entrees))



encoder_input_data = np.zeros(
    (len(entrees), max_encoder_seq_length, taille_vec_mot), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(entrees), max_decoder_seq_length, taille_vec_mot), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(entrees), max_decoder_seq_length, taille_vec_mot), dtype="float32"
)


#il encode en one hot chac par char
#/!\ 



print("Génération des données d'entrainement")

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, mot in enumerate(input_text):
        encoder_input_data[i, t] = entrees[i,t]
    for t, mot in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep 
        decoder_input_data[i, t] = entrees[i,t] # je crois que ça ne sert qu'a ininitialiser, donc on peut mettre rnd ???? (jsp)
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character. 
            decoder_target_data[i, t - 1] = entrees[i,t] # je crois que ça ne sert qu'a ininitialiser, donc on peut mettre rnd ???? (jsp)

print('Génération du modèle')



# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, taille_vec_mot))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, taille_vec_mot))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(taille_vec_mot, activation="sigmoid")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="rmsprop", loss="mean_squared_error", metrics=["accuracy"]
)
#model.compile(
#    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
#)

print("Entrainement")

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
# Save model
#model.save("s2sae")



# Define sampling models
# Restore the model and construct the encoder and decoder.
#model = keras.models.load_model("s2sae")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)




def decode_sequence(input_seq):
    vecteurs_sortie =[]
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.array([[skipgram.wv["<begin>"]]])


    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    nb_mot =0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        vecteurs_sortie.append(output_tokens[0][0])
        # Sample a token
        mot_decode = skipgram.wv.similar_by_vector(output_tokens[0][0],topn=1)[0][0]
        # Exit condition: either hit max length
        # or find stop character.
        if mot_decode == "<end>" or nb_mot > max_decoder_seq_length:
            stop_condition = True
        else:
            # Update the target sequence (of length 1).
            decoded_sentence += mot_decode + " "
            nb_mot += 1
            target_seq = np.array([[skipgram.wv[mot_decode]]])


        # Update states
        states_value = [h, c]
    return decoded_sentence, vecteurs_sortie


print("Des tests")

for seq_index in range(1):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence,vecteurs_sortie = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", lemmatizedSentencesSpacy[seq_index])
    print("Decoded sentence:", decoded_sentence)
    #print("vecteurs entrée:", input_seq[0])
    #print("Vecteurs_sorti:", vecteurs_sortie)

