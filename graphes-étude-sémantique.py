# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:31:54 2022

@author: matth
"""
import gensim
import nltk
import matplotlib
import spacy
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.callbacks import CallbackAny2Vec 
from spacy.lang.fr.stop_words import STOP_WORDS as stopwordsSpacy
import re as regex
import pandas as pd
import scipy

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

print('bibliothèques importées\n')

nlp = spacy.load('fr_core_news_sm')
print('modèle spacy importé\n')

path = './word2vec_docs_scol_traités/'
path_fig = './word2vec_docs_scol_traités/figures-rapport-études-sémantique/'

with open(path+'corpus.txt', 'r', encoding="utf-8") as f:
    text = f.read()
    
    
text = text.replace("\n", " ")


phrases = nltk.tokenize.sent_tokenize(text, language='french')
print('phrases parsées par NLTK\n')
phrasesTokeniseesSpacy = [nlp(s) for s in phrases]
print('phrases tokénisées par spacy\n')

phrasesDecoupeesSpacy = [[token.text.lower() for token in doc] for doc in phrasesTokeniseesSpacy]
print('phrases découpées en tokens\n')


class callback(CallbackAny2Vec): # pour avoir un rendu verbose du word2vec
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

#%% WORD2VEC



def entropieMoyenne(vecteurs):
    H = 0
    st_dev = [np.std(vecteurs[:,d])     for d in range(vecteurs.shape[1])]
    moy    = [np.average(vecteurs[:,d]) for d in range(vecteurs.shape[1])]
    total_std = np.sum(st_dev)
    for d in range(vecteurs.shape[1]):
        histo = np.histogram(vecteurs[:,d],bins=50)
        freqs = histo[0]/vecteurs.shape[0]
        H += scipy.stats.entropy(freqs)*st_dev[d]**2
    return vecteurs.shape[1]*H/total_std





print('word2vec achevé\n')

def modele2vecteursWV(model,dimension,nbmax=-1):
    '''renvoie
            la liste des nbmax premiers mots sous la forme d'une liste
            et leurs vecteurs sous la forme d'un array'''
    whitelist = regex.compile('[^ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzôûâîàéêçèùÉÀÈÇÂÊÎÔÛÄËÏÖÜÀÇÉÈÙ]')
    mots = [word for word in model.wv.index_to_key if whitelist.search(word) is None and word not in stopwordsSpacy]
    vecteurs = np.empty((len(mots), dimension), dtype='f')
    for i in range(len(mots)):
        vecteurs[i] = model.wv[mots[i]]
    return mots[:nbmax], vecteurs[:nbmax]





def visualiserMotsFrequentsCategorises(model,nb_quiver=70,pca_components=10,nbcat=5):
    mots_selectionnes, vecteurs = modele2vecteursWV(model,dimension,nb_quiver)
    # Trouver les coordonnées de  t-SNE en deux dimensions
    np.set_printoptions(suppress=True)        
    # Réduire la domention de 300 à 10 avec PCA
    reduc = PCA(n_components=pca_components).fit_transform(vecteurs)
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    kmeans = KMeans(n_clusters=nbcat)
    kmeans.fit(vecteurs)
    categories = kmeans.predict(vecteurs)
    fig = plt.figure(edgecolor='black', dpi=400)
    ax = fig.add_subplot(111)
    ax.scatter(Y[:, 0], Y[:, 1], c=categories, s=40, cmap='rainbow', marker=".")
    ax.set_aspect('equal')
    for i in range(np.shape(Y)[0]):
        ax.text(Y[i][0],Y[i][1], mots_selectionnes[i], size=5)
    plt.title(f'Les {nb_quiver} mots les plus fréquents sur {len(model.wv)}\nDimension départ {dimension} +PCA -> {pca_components} + t-SNE -> 2\n{"Skip-gram" if model.sg==1 else "CBOW"} fenêtre {taille_fenetre}, {iterations} itérations', size='medium')
    plt.savefig(path_fig+f'word2vec_pca_dim{dimension}.pdf', bbox_inches='tight')
    plt.show()
    
def visualiserCorrelationMots(model,nb_cor=15):
    mots_selectionnes, vecteurs = modele2vecteursWV(model,dimension,nb_cor)
    for i in range(nb_cor):
        vecteurs[i] = vecteurs[i]/np.linalg.norm(vecteurs[i])

    mat_corr_np = np.dot(vecteurs[:nb_cor],np.transpose(vecteurs[:nb_cor])) 

    plt.figure(figsize = (8,8))
    plt.imshow(mat_corr_np,cmap='GnBu')
    plt.colorbar()
    plt.grid(False)
    plt.xticks(range(nb_cor), mots_selectionnes[:nb_cor], rotation=65, fontsize=10)
    plt.yticks(range(nb_cor), mots_selectionnes[:nb_cor], rotation='horizontal', fontsize=10)
    plt.title(f'Matrice de corrélation des \n{nb_cor} mots les plus fréquents\n{dimension} dimensions; {"Skip-gram" if model.sg==1 else "CBOW"}; fenêtre {taille_fenetre}; {iterations} itérations', size=15)
    plt.savefig(path_fig+f'matrice_correlation_dim{dimension}.pdf',bbox_inches='tight')
    plt.show()

def visualiserVecteursGraphes(model,nb_vec=15):
    mots_selectionnes, vecteurs = modele2vecteursWV(model,dimension,nb_vec)
    for i in range(nb_vec):
        vecteurs[i] = vecteurs[i]/np.linalg.norm(vecteurs[i])

    plt.figure(figsize = (12,8),dpi=400)
    for i in range(nb_vec):
        plt.plot(vecteurs[i])
    plt.grid(False)
    plt.title(f'Vecteurs des {nb_vec} mots les plus fréquents\n{dimension} dimensions; {"Skip-gram" if model.sg==1 else "CBOW"}; fenêtre {taille_fenetre}; {iterations} itérations', size=15)
    plt.savefig(path_fig+f'vecteurs_graphes_dim{dimension}.pdf',bbox_inches='tight')
    plt.show()


def visualiserValeursDimensions(model,nb_vec):
    _, vecteurs = modele2vecteursWV(model,dimension,nb_vec)

    fig, axs = plt.subplots(nrows=2, ncols=2,figsize = (12,8))
    
    groupes_dim = np.array_split(range(dimension), 4)
    for dims_l,ax in zip(groupes_dim,axs.reshape(-1)):
        for d in dims_l:
            histo = np.histogram(vecteurs[:,d],bins=50)
            freqs = histo[0]/vecteurs.shape[0]
            entropy = scipy.stats.entropy(freqs)
            ax.fill_between(histo[1][:-1], histo[0], step="post", alpha=0.4, label = f'dim{d+1}: H={round(entropy,3)}')
        ax.grid(True)    
        ax.legend()
        ax.set_xlabel('valeur sur l\'axe de la dimension')
        ax.set_ylabel('nombre de mots')
    fig.suptitle(f'Distribution des valeurs sur les {vecteurs.shape[0]} premiers mots les plus fréquents de l\'embedding\n50 classes; {dimension} dimensions; {"Skip-gram" if model.sg==1 else "CBOW"} fenêtre {taille_fenetre}; {iterations} itérations', size=15)
    plt.savefig(path_fig+f'distributions_dimensions_dim{dimension}.pdf',bbox_inches='tight')
    plt.show()
    


def visualiserCorrelationsDimensions(model):
    _, vecteurs = modele2vecteursWV(word2vec,dimension)
    
    for i in range(vecteurs.shape[0]):
        vecteurs[i] = vecteurs[i]/np.linalg.norm(vecteurs[i])
    plt.figure(dpi=400)
    mat_cor = np.cov(vecteurs.T)
    plt.matshow(mat_cor,cmap='RdBu', vmin=-.02, vmax=.02)
    plt.colorbar()
    plt.title(rf'Corrélation des dimensions '+f'\n{dimension} dimensions; {"Skip-gram" if model.sg==1 else "CBOW"} fenêtre {taille_fenetre}; {iterations} itérations')
    plt.savefig(path_fig+f'correlation_dimensions_dim{dimension}.pdf',bbox_inches='tight')
    plt.show()
    
    
def visualiserDivergence(model):
    _, vecteurs = modele2vecteursWV(model,dimension)
    l_histo = [np.histogram(vecteurs[:,d],bins=10)[0]/vecteurs.shape[0] for d in range(vecteurs.shape[1])]
    cross_ent = np.zeros((vecteurs.shape[1],vecteurs.shape[1]),dtype=float)
    for d1 in range(vecteurs.shape[1]):
        for d2 in range(vecteurs.shape[1]):
            cross_ent[d1,d2] = scipy.stats.entropy(l_histo[d1],l_histo[d2])
    plt.matshow(cross_ent)
    plt.colorbar()
    return cross_ent


 
    for i in range(vecteurs.shape[0]):
        vecteurs[i] = vecteurs[i]/np.linalg.norm(vecteurs[i])
    plt.figure(dpi=400)
    mat_cor = np.cov(vecteurs.T)
    plt.matshow(mat_cor,cmap='RdBu', vmin=-.02, vmax=.02)
    plt.colorbar()
    plt.title(rf'Corrélation des dimensions '+f'\n{dimension} dimensions; {"Skip-gram" if model.sg==1 else "CBOW"} fenêtre {taille_fenetre}; {iterations} itérations')
    plt.savefig(path_fig+f'correlation_dimensions_dim{dimension}.pdf',bbox_inches='tight')
    plt.show()

def visualiserDistributionsDimensions(model,nb_dim):
    _, vecteurs = modele2vecteursWV(word2vec,nb_dim)
    plt.boxplot(vecteurs,showfliers=False)
    plt.grid(True)
    plt.title(f'Données statistique des dimensions\n{dimension} dimensions; {"Skip-gram" if model.sg==1 else "CBOW"} fenêtre {taille_fenetre}; {iterations} itérations')
    plt.savefig(path_fig+f'boxgraph_dimensions_dim{dimension}.pdf',bbox_inches='tight')
    plt.show()
    
    

dimension = 32
taille_fenetre = 5
iterations = 25

# 1 pour skipram, 0 pour cbow
mode=1

word2vec = gensim.models.Word2Vec(sentences=phrasesDecoupeesSpacy, window=taille_fenetre, min_count=1, sg=mode, vector_size=dimension, workers=4,epochs=iterations,callbacks=[callback()]) 

taille_vocab=len(word2vec.wv)
mots_selectionnes, vecteurs = modele2vecteursWV(word2vec,dimension)

print(dimension, entropieMoyenne(vecteurs))

    

visualiserMotsFrequentsCategorises(word2vec,nb_quiver=100,pca_components=8,nbcat=6)
visualiserCorrelationMots(word2vec,nb_cor=30)
visualiserVecteursGraphes(word2vec,nb_vec=100)
visualiserValeursDimensions(word2vec,nb_vec=taille_vocab)
visualiserCorrelationsDimensions(word2vec)
visualiserDistributionsDimensions(word2vec,dimension)

mots_selectionnes, vecteurs = modele2vecteursWV(word2vec,dimension)
for i in range(vecteurs.shape[-1]):
    vecteurs[i] = vecteurs[i]/np.linalg.norm(vecteurs[i])
df = pd.DataFrame(vecteurs,columns = [f"dim{n}" for n in range(1,dimension+1)])



