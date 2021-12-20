# -*- coding: utf-8 -*-



import gensim
import nltk
import string
from gensim.models import Word2Vec
from sklearn.decomposition import PCA


from nltk.tokenize import word_tokenize, sent_tokenize


path = './documents_scolarite/'

texte = ''

documents = ['consignes-mobilite-covid19.txt', 'dispense-de-mobilite-obligatoire.txt', 'mobilite-3a-explication-de-la-procedure-intranet.txt', 'mobilite-cesure-choix-post-2a.txt', 'procedure-selection-ceu.txt', 'reglement-scolarite-2021.txt', 'sejour-international.txt', 'validation-post-tfe.txt']

for i in range(len(documents)):
    documents[i] = path+documents[i]
    f = open(documents[i], 'r', encoding="utf-8")
    texte += f.read()
    f.close()


with open(path+'total.txt', 'w', encoding="utf-8") as f:
    f.write(texte)
    f.close()






phrases = texte.split("\n")
phrases = [c for c in phrases if len(c) != 0]





def gensim_preprocess_data(texte):
    stopwords = set(nltk.corpus.stopwords.words('french'))
    stopWordsFrench = stopwords | set([l[0].upper()+l[1:] for l in stopwords])
    sentences = sent_tokenize(texte)
    tokenized_sentences = list([word_tokenize(sentence) for sentence in sentences])
    for i in range(0, len(tokenized_sentences)):
        tokenized_sentences[i] = [word for word in tokenized_sentences[i] if word not in string.punctuation and word not in stopWordsFrench]
    return tokenized_sentences

phrases = gensim_preprocess_data(texte)


dimension = 300

skipgram = Word2Vec(sentences=phrases, window=5, min_count=1, sg=1, vector_size=dimension) # 1 pour skip gram, 0 pour cbow

#word_embedding = skip_gram[skip_gram.wv.vocab] 




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# code inspiré du turotiel de Gensim sur kaggle
def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, dimension), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 40 to 10 dimensions with PCA
    reduc = PCA(n_components=2).fit_transform(arrays)
    
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in reduc[:, 0]],
                       'y': [y for y in reduc[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(reduc[:, 0].min(), reduc[:, 0].max())
    plt.ylim(reduc[:, 1].min(), reduc[:, 1].max())
            
    plt.title('visualisation t-SNE for {}'.format(word.title()))
    
tsnescatterplot(skipgram, 'Centrale', [])
    
    
"""
X = skipgram[skipgram.wv]
pca = PCA(n_components=2)
result = pca.fit_transform(X)


# pour accéder au vecteur du mot centrale
skipgram.wv["Centrale"]
skipgram.wv.key_to_index # pour accéder au dictionnaire des mots
skipgram.wv.wmdistance(["procédure"],["scolarité"])
skipgram.wv.most_similar([word])
"""