# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 10:30:12 2022

@author: matth
"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import nltk
import spacy
import re
import gensim
import os

print('bibliothèques importées')

with open("./Clement's Lab/PhrasesFR.txt", 'r', encoding='utf-8') as f:
    texte = f.read()

print('texte importé')


nb_max_tokenized_sentences = 10000
embedding_dim = 64
latent_dim = 128


sentences = nltk.tokenize.sent_tokenize(texte, language='french')[:nb_max_tokenized_sentences]

nlp = spacy.load('fr_core_news_sm')
print('texte segmenté par NLTK')



def preprocess_sentence(w):
    '''ramener en minuscules, enlever les unicode non autorisés, rajouter des espaces après les ponctuations'''
    w = w.lower()
    w = re.sub(r'[^ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäëïöüâêîôûàéêèçùÉÀÈÇÂÊÎÔÛÄËÏÖÜÀÇÉÈÙ,:.?!;\'-]',' ', w)
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = "begin " + w + " end"
    w = re.sub(r'[" "]+', " ", w) # replace several spaces by only one
    return w


sentences = [preprocess_sentence(w) for w in sentences]  # prétraitement

tokenized_sentences = [[tok.text for tok in nlp(sent)] for sent in sentences] # tokénisation



reconstructed_sentences = [' '.join(sent) for sent in tokenized_sentences]

print('phrases tokénisées par spacy')


from gensim.models import Word2Vec
skipgram = Word2Vec(sentences=tokenized_sentences, window=5, min_count=1, sg=1, vector_size=embedding_dim, workers=4,epochs=10)
sentence_max_length = max([len(sent) for sent in tokenized_sentences])
vocab_size = len(skipgram.wv.index_to_key)




from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(filters='')
# Convert sequences into internal vocab
tokenizer.fit_on_texts(reconstructed_sentences)
# Convert internal vocab to numbers
tensor = tokenizer.texts_to_sequences(reconstructed_sentences)
# Pad the tensors to assign equal length to all the sequences
tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')


'''À ce stade on a len(skipgram.wv.index_to_key) = len(tokenizer.word_index)
Mais la taille du vocabulaire utilisé va être incrémentée de 1 dans le code pourquoi ? pas compris, sûrement pour les mots inconnus et encore ils ne sont pas gérés dans la conversion de la  fonction de traduction finale


on a tokenizer.index_word[i] = skipgram.wv.index_to_key[i-1]
cf erreur "Layer embedding weight shape (247, 128) is not compatible with provided weight shape (246, 128)" lorsqu'on passe l'argument weights=[pretrained_weights] à l'embedding,  il faut donner en entrée des poids une matrice de taille de voc + 1'''


pretrained_weights = np.zeros((vocab_size+1,embedding_dim),dtype=np.float32)
# pretrained_weights[1:][:] = skipgram.syn1neg les premiers mots coïncident mais pas les derniers, pourquoi ??
for word in list(iter(tokenizer.word_index)):
    idx = skipgram.wv.key_to_index[word]
    pretrained_weights[idx] = skipgram.wv[word]




# Show the mapping b/w word index and language tokenizer
def convert(tokenizer, tensor):
    for t in tensor: # t est un entier élément du tenseur
        if t != 0:
            print ("%d ----> %s" % (t, tokenizer.index_word[t]))

convert(tokenizer,tensor[25])
print(reconstructed_sentences[25])


# separate in validation and train
from sklearn.model_selection import train_test_split
input_tensor_train, input_tensor_valid, target_tensor_train, target_tensor_valid = train_test_split(tensor, tensor, test_size=0.2)



# Essential model parameters
BUFFER_SIZE = len(input_tensor_train) # equivalent to selecting first dimension
BATCH_SIZE = 16
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE # = num_examples//BATCH_SIZE
# embedding_dim = 128
# latent_dim = 128

vocab_inp_size = len(tokenizer.word_index) + 1 # pourquoi ?
vocab_tar_size = len(tokenizer.word_index) + 1


dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

print("\ncharacteristics of dataset")
print(type(dataset), len(dataset))
g = list(dataset.as_numpy_iterator())

print("\ncharacteristics of list(dataset.as_numpy_iterator())")

print(type(g),len(g), "(= steps_per_epoch)")
print(type(g[0]), len(g[0])) 
print(type(g[0][0]), np.shape(g[0][0]), "(= BATCH_SIZE, sentence_max_length)")


# Encoder class
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_latent_dim, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_latent_dim = enc_latent_dim

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[pretrained_weights])
    self.gru = tf.keras.layers.GRU(self.enc_latent_dim,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state=hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_latent_dim))




# Attention Mechanism
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, latent_dim):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(latent_dim)
    self.W2 = tf.keras.layers.Dense(latent_dim)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # values shape == (batch_size, max_len, hidden size)

    # we are doing this to broadcast addition along the time axis to calculate the score
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, latent_dim)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights



# Decoder class
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_latent_dim, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_latent_dim = dec_latent_dim
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_latent_dim,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Used for attention
    self.attention = BahdanauAttention(self.dec_latent_dim)

  def call(self, x, hidden, enc_output):
    # x shape == (batch_size, 1)
    # hidden shape == (batch_size, max_length)
    # enc_output shape == (batch_size, max_length, hidden_size)

    # context_vector shape == (batch_size, hidden_size)
    # attention_weights shape == (batch_size, max_length, 1)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights






encoder = Encoder(vocab_inp_size, embedding_dim, latent_dim, BATCH_SIZE)

print("\n*** Size of input and target batches ***")
example_input_batch, example_target_batch = next(iter(dataset)) # sélectionner un batch
print("Input  batch: (batch size, sequence_length) {}".format(example_input_batch.shape))
print("Target batch: (batch size, sequence_length) {}".format(example_target_batch.shape))



print("\n*** Size of ENDODER's output and state ***")
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print("Encoder output: {}".format(sample_output.shape))
print("Encoder hidden: {}".format(sample_hidden.shape))



print("\n*** Size of ATTENTION's output ***")
attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
print("Attention result shape: (batch size, latent_dim) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


print("\n*** Size DECODER's output ***")
decoder = Decoder(vocab_tar_size, embedding_dim, latent_dim, BATCH_SIZE)
sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),sample_hidden, sample_output)
print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))






optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    # Take care of the padding. Not all sequences are of equal length.
    # If there's a '0' in the sequence, the loss is being nullified
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    # tf.GradientTape() -- record operations for automatic differentiation
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        # dec_hidden is used by attention, hence is the same enc_hidden
        dec_hidden = enc_hidden
        # <start> token is the initial decoder input
        dec_input = tf.expand_dims([tokenizer.word_index['begin']] * BATCH_SIZE, 1)
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # Pass enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            # Compute the loss
            loss += loss_function(targ[:, t], predictions)
            # Use teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)
    # As this function is called per batch, compute the batch_loss
    batch_loss = (loss / int(targ.shape[1]))
    # Get the model's variables
    variables = encoder.trainable_variables + decoder.trainable_variables
    # Compute the gradients
    gradients = tape.gradient(loss, variables)
    # Update the variables of the model/network
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss



import time

EPOCHS = 50

# Training loop
for epoch in range(EPOCHS):
    start = time.time()
    
    # Initialize the hidden state
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    
    # Loop through the dataset
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
      
        # Call the train method
        batch_loss = train_step(inp, targ, enc_hidden)
      
        # Compute the loss (per batch)
        total_loss += batch_loss
      
        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
    # Save (checkpoint) the model every 2 epochs
    if epoch == EPOCHS-1:
        checkpoint.save(file_prefix = checkpoint_prefix)
    
    # Output the loss observed until that epoch
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    
    
# Evaluate function -- similar to the training loop
def evaluate(sentence):
    
    # Attention plot (to be plotted later on) -- initialized with max_lengths of both target and input
    attention_plot = np.zeros((sentence_max_length, sentence_max_length))
    
    # Preprocess the sentence given
    sentence = preprocess_sentence(sentence)
    
    # Fetch the indices concerning the words in the sentence and pad the sequence
    inputs = [tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=sentence_max_length,
                                                           padding='post')
    # Convert the inputs to tensors
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''
    
    hidden = [tf.zeros((1, latent_dim))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tokenizer.word_index['begin']], 0)
    
    # Loop until the max_length is reached for the target lang (ENGLISH)
    for t in range(sentence_max_length):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
      
        # Store the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
      
        # Get the prediction with the maximum attention
        predicted_id = tf.argmax(predictions[0]).numpy()
      
        # Append the token to the result
        result += tokenizer.index_word[predicted_id] + ' '
      
        # If <end> token is reached, return the result, input, and attention plot
        if tokenizer.index_word[predicted_id] == 'end':
          return result, sentence, attention_plot
      
        # The predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    
    return result, sentence, attention_plot


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.show()

# Translate function (which internally calls the evaluate function)
def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)
    
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))
    
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


translate(u"ou es tu maintenant ?")

