# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 21:46:11 2021

@author: cleme
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import gensim

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 252  # Latent dimensionality of the encoding space.
num_samples = 1000  # Number of samples to train on.
taille_vec_mot = 64
# Path to the data txt file on disk.
data_path = "PhrasesAng.txt"

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()




#traitement des données

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    f.close()
processedLines = [gensim.utils.simple_preprocess(sentence) for sentence in lines]
word_list = [word for words in processedLines for word in words]
   
for phrase in processedLines[: min(num_samples, len(lines) - 1)]:
    
    #ce code génère les données d'entrée sorties voulues en input_text et target text
    
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    phrase_cible = ["<begin>"] +phrase + ["<end>"] #la cible est donc l'entrée 
    input_texts.append(phrase)
    target_texts.append(phrase_cible)
    


max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

print("Generation du Word2Vec")




#check the length of the list
print('Nombre de mots ', len(word_list))

modelv = gensim.models.Word2Vec(
        [word_list],
        vector_size=taille_vec_mot,
        min_count=1,
        workers=4)

#ajout des mots d'entrée et de fin
modelv.wv.add_vector('<begin>',np.ones(taille_vec_mot))
modelv.wv.add_vector('<end>',np.zeros(taille_vec_mot))



#input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
#target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])






#liste de liste de liste: liste des phrases à apprednre; phrase a apprendre : liste de taille max_taille_liste_entree de vecteurs des caractères

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, taille_vec_mot), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, taille_vec_mot), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, taille_vec_mot), dtype="float32"
)


#il encode en one hot chac par char
#/!\ 



print("Génération des données d'entrainement")

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, mot in enumerate(input_text):
        encoder_input_data[i, t] = modelv.wv[mot]
    for t, mot in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep 
        decoder_input_data[i, t] = modelv.wv[mot] # je crois que ça ne sert qu'a ininitialiser, donc on peut mettre rnd ???? (jsp)
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character. 
            decoder_target_data[i, t - 1] = modelv.wv[mot] # je crois que ça ne sert qu'a ininitialiser, donc on peut mettre rnd ???? (jsp)

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
    optimizer="rmsprop", loss="MeanSquaredError", metrics=["accuracy"]
)

print("Entrainement")

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
# Save model
model.save("s2sae")



# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model("s2sae")

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
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)


    # Generate empty target sequence of length 1.
    target_seq = np.array([[modelv.wv["<begin>"]]])


    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    nb_mot =0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        mot_decode = modelv.wv.similar_by_vector(output_tokens[0][0],topn=1)[0][0]
        # Exit condition: either hit max length
        # or find stop character.
        if mot_decode == "<end>" or nb_mot > max_decoder_seq_length:
            stop_condition = True
        else:
            # Update the target sequence (of length 1).
            decoded_sentence += mot_decode + " "
            nb_mot += 1
            target_seq = np.array([[modelv.wv[mot_decode]]])


        # Update states
        states_value = [h, c]
    return decoded_sentence


print("Des tests")

for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)

