# -*- coding: utf-8 -*-



import gensim

import nltk
import matplotlib

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import re as regex
import string
import spacy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
         
import seaborn as sns
sns.set_style("darkgrid")


print('\nimportation des bibliothèques achevée\n')
        


# création des ensembles de stopwords




# librement inspiré du tutpo Gensim sur Kaggle
class WordEmbedding():
    def __init__(self, chemin):
        self.text = None
        self.path = chemin
        self.dimension = None
        self.win_size = None
        self.mode = None
        self.stopwordsSpacy = spacy.lang.fr.stop_words.STOP_WORDS
        self.stopwordsNLTK = set(nltk.corpus.stopwords.words('french'))
        

    def importerCorpus(self,nomfichier):
        self.text = ''
        
        with open(self.path+nomfichier, 'r', encoding="utf-8") as f:
            self.text = f.read()
        self.text = self.text.replace("\n", " ")
        

    ## NLTK
    def segmenterTexteNLTK(self):
        '''segmente le texte en phrases avec NLTK
           renvoie une liste d'éléments de type string'''
        return nltk.tokenize.sent_tokenize(self.text, language='french')  

        
    def tokeniserPhrasesNLTK(self,phrases):
        '''prend en entrée phrases: liste de strings
           tokénise les phrases en tokens avec NLTK
           renvoie une liste d'élements de type string'''
        return [nltk.tokenize.word_tokenize(s, language='french') for s in phrases]
    
    ## SPACY
    def definirModeleSpacy(self, model='fr_core_news_sm'):
        ''' importe un modèle de langage prédéfini de spacy
            fr_core_news_sm pour efficacité
            fr_dep_news_trf précision (mais hyper lent et gourmand en mémoire)'''
        self.nlp = spacy.load(model)

    
    def segmenterTexteSpacy(self):
        '''segmente le texte en phrases avec Spacy
           renvoie une liste d'élements de type spacy.tokens.span.Span'''
        doc = self.nlp(self.text)
        return [s for s in doc.sents]
    
    def tokeniserPhrasesSpacy(self,phrases):
        '''prend en entrée phrases: liste de strings
           tokénise les phrases en tokens avec Spacy
           renvoie une liste d'élements de type spacy.tokens.doc.Doc
           chaque élement de la liste est un itérable qui se scinde
           en éléments de type spacy.tokens.token.Token'''
        return [self.nlp(s) for s in phrases]
    
    
    def visualiserArbreDependancePhrase(self,s):
        '''renvoie l'arbre de dépendance d'une phrase s de type string'''
        doc = self.nlp(s)
        spacy.displacy.serve(doc, style="dep")
        
    def visualiserTokensPhrase(self,s):
        for token in self.nlp(s):
            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop, token.morph, token.sentiment, token. ent_type, token.ent_type_, token.dep)
    
    
    def enregistrerPhrases(self,l,nomfichier):
        '''l est une liste d'éléments' de type list ou spacy.tokens.doc.Doc'''
        with open(self.path+nomfichier, 'w', encoding='utf8') as f:
            for elt in l:
                f.write(f'{elt.text}' + "\n\n") if type(elt) == spacy.tokens.doc.Doc else f.write(str(elt) + "\n\n")
            
            
    def lemmatiserPhrasesSpacy(self,phrases):
        '''phrases est une liste de phrases de type spacy.tokens.doc.Doc
           renvoie une liste de listes de strings donc chaque string est un mot
           en minuscules, lemmatisé si ce n'est pas une entité
           la ponctuation est retirée'''
        lemmatizedSentencesSpacy = []
        for p in phrases:
            explodedSentence = []
            for token in p:
                if not token.is_punct:
                    if token.ent_type_ != '': # déterminer si le token fait partie d'une entité
                        explodedSentence.append(token.text.lower()) # if token.text.lower() not in self.stopwordsSpacy else None
                    else:
                        explodedSentence.append(token.lemma_.lower()) # if token.text.lower() not in self.stopwordsSpacy else None
            lemmatizedSentencesSpacy.append(explodedSentence)
        return lemmatizedSentencesSpacy
        
   
    def simplifierPhrasesSpacy(self,phrases):
        '''phrases est une liste de phrases de type spacy.tokens.doc.Doc
           renvoie une liste strings donc chaque string est une phrases
           en minuscules, lemmatisé sans ponctuation et sans stopwords'''
        simplifiedSentencesSpacy = []
        for p in phrases:
            explodedSentence = []
            for token in p:
                if not token.is_punct:
                    explodedSentence.append(token.lemma_.lower()) # if token.text.lower() not in self.stopwordsSpacy else None
            simplifiedSentencesSpacy.append(' '.join(explodedSentence))
        return simplifiedSentencesSpacy
    
    
    
    def word2vec(self,phrases,dimension=250,taille_fenetre=6,mode=1):
        '''phrases doit être une liste de liste de strings dont chacun est un mot
           mode 1 pour skip gram, 0 pour cbow'''
        self.dimension = dimension
        self.win_size = taille_fenetre
        self.mode = 'Skip-Gram' if mode == 1 else 'CBOW'
        return gensim.models.Word2Vec(sentences=phrases, window=taille_fenetre, min_count=1, sg=mode, vector_size=dimension, workers=4) 
  
            
        
    def visualiserMotsFrequentsCategorises(self,model,nbmax=150,pca_components=10,nbcat=5):
        mots_selectionnes, vecteurs = self.modele2vecteurs(model,nbmax)
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
        plt.title(f'Visualisation des {nbmax} mots les plus fréquents sur {len(model.wv)}\nDimension départ {self.dimension} +PCA -> {pca_components} + t-SNE -> 2\nSkip-gram fenêtre {self.win_size}', size='medium')


    
    
    def visualiserMotsFrequents(self,model,nbmax,pca_components):  
        mots_selectionnes, vecteurs = self.modele2vecteurs(model,nbmax)
        color_list  = len(mots_selectionnes) * ['blue']
        # Trouver les coordonnées de  t-SNE en deux dimensions
        np.set_printoptions(suppress=True)        
        # Réduire la domention de 300 à 10 avec PCA
        reduc = PCA(n_components=pca_components).fit_transform(vecteurs)
        
        Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    
        # Fabriquer le cadre pour le plot
        df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                           'y': [y for y in Y[:, 1]],
                           'words': mots_selectionnes,
                           'color': color_list})
        
        fig, _ = plt.subplots()
        fig.set_size_inches(9, 9)
        
        # Faire le plot des mots
        p1 = sns.regplot(data=df,
                         x="x",
                         y="y",
                         fit_reg=False,
                         marker="o",
                         scatter_kws={'s': 50,
                                      'facecolors': df['color']
                                     }
                        )
        
        # Ajouter les annotations avec une boucle
        for line in range(0, df.shape[0]):
             p1.text(df["x"][line],
                     df['y'][line],
                     '  ' + df["words"][line].title(),
                     horizontalalignment='left',
                     verticalalignment='bottom', size='xx-small',
                     color=df['color'][line],
                     weight='normal'
                    ).set_size(8)
        
        plt.xlim(Y[:, 0].min()*1.05, Y[:, 0].max()*1.05)
        plt.ylim(Y[:, 1].min()*1.05, Y[:, 1].max()*1.05)
                
        plt.title(f'Visualisation des {nbmax} mots les plus fréquents sur {len(model.wv)}\nDimension départ {self.dimension} +PCA -> {pca_components} + t-SNE -> 2\nSkip-gram fenêtre {self.win_size}', size='medium')
        
        
    def modele2vecteurs(self,model,nbmax=-1):
        '''renvoie la liste des nbmax premiers mots sous la forme d'une liste
        et leurs vecteurs sous la forme d'un array'''
        whitelist = regex.compile('[^ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzôûâîàéêçèùÉÀÈÇÂÊÎÔÛÄËÏÖÜÀÇÉÈÙ]')
        mots = [word for word in model.wv.index_to_key if whitelist.search(word) is None and word not in self.stopwordsSpacy]
        vecteurs = np.empty((len(mots), self.dimension), dtype='f')
        for i in range(len(mots)):
            vecteurs[i] = model.wv[mots[i]]
        return mots[:nbmax], vecteurs[:nbmax]
            
    
    def phrases2tenseur(self,model,phrases):
        '''renvoie un array numpy de dimension
        nb_phrases x nb_max_mots x dim_embedding'''
        tenseurPhrases = np.zeros((len(phrases),max([len(p) for p in phrases]), self.dimension))
        for p in range(len(phrases)):
            for m in range(len(phrases[p])):
                tenseurPhrases[p][m][:] = model.wv[phrases[p][m]]
        return tenseurPhrases


    def enregistrerTenseurPhrases(self,tenseur,fichier):
        '''enregistre l'array numpy des vecteurs de toutes les phrases'''
        np.save(self.path+fichier,tenseur)
        
            
    def enregistrerVecteursVocabulaire(self,model,vocabulaire,fichier_vocabulaire,vecteurs,fichier_vecteurs):
        '''enregistre les mots dans le fichier 'vocabulaire.txt'
        et leurs vecteurs dans le fichier vecteurs.npy (en binaire numpy'''
        np.save(fichier_vecteurs, vecteurs)
        with open(fichier_vocabulaire, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocabulaire))
    
    
if __name__ == "__main__":
    path = './word2vec_docs_scol_traités/'
    wr = WordEmbedding(path)
    print("importation du corpus\n")
    wr.importerCorpus('corpus.txt')
    print("commencement segmentation+tokénisation+lemmatisation\n")
    sentences = wr.segmenterTexteNLTK()
    print("importation du modèle spacy")
    wr.definirModeleSpacy('fr_core_news_sm') #fr_core_news_sm erreur cursus fr_core_news_md fr_dep_news_trf
    print("tokénisation des phrases")
    phrases = wr.segmenterTexteNLTK() # semble plus rapide et moins gourmand que Spacy
    phrasesTokeniseesSpacy = wr.tokeniserPhrasesSpacy(phrases)
    wr.enregistrerPhrases(phrasesTokeniseesSpacy,'phrases.txt')
    print("phrases tokénisées et enregistrées !\n")
    
    print('lemmatisation des phrases\n')
    phrasesLemmatisees = wr.lemmatiserPhrasesSpacy(phrasesTokeniseesSpacy)
    phrasesSimplifiees = wr.simplifierPhrasesSpacy(phrasesTokeniseesSpacy)
    phrasesSimplifieesEclatees = [p.split(' ') for p in phrasesSimplifiees]

    
    print('enregistrement des phrases lemmatisées\n')
    wr.enregistrerPhrases(phrasesLemmatisees,'phrases-lemmatisées.txt')
    
    print('enregistrement phrases simplifiées et vecteurs\n')
    wr.enregistrerPhrases(phrasesSimplifiees,'phrases-simplifiées.txt')
    
    

    dimension = 256
    print('commencement word2vec\n')
    skipgram = wr.word2vec(phrasesSimplifieesEclatees,dimension=dimension,taille_fenetre=6,mode=1)
    print("word2vec achevé\n")
    skipgram.save(path+"skipgram.model")
    
    # wr.visualiserMotsFrequents(skipgram,50,10)
    # print(phrasesTokeniseesSpacy[15],'\n')
    # wr.visualiserTokensPhrase(phrasesTokeniseesSpacy[15])
    # wr.visualiserArbreDependancePhrase(phrasesTokeniseesSpacy[15].text)
    
    vocabulaire,vecteurs = wr.modele2vecteurs(skipgram)
    wr.enregistrerVecteursVocabulaire(skipgram,vocabulaire,'vocabulaire.txt',vecteurs,'vecteurs.npy')
    print('enregistrement vocabulaire et vecteurs\n')
    
    
    phrasesSimplifieesEclatees = [p for p in phrasesSimplifieesEclatees if len(p) <= 100] # ne sélectionner que les phrases les plus courtes
    
    tenseurPhrases = wr.phrases2tenseur(skipgram,phrasesSimplifieesEclatees) # le second argument doit correspondre aux phrases données pour la création du word2vec
    wr.enregistrerTenseurPhrases(tenseurPhrases,'phrases-vecteurs.npy')
    
    pca = 8
    nbcat = 4
    nbmots = 70
    wr.visualiserMotsFrequentsCategorises(skipgram,nbmots,pca,nbcat)
    # plt.savefig(f'./word2vec-dim{dimension}-pca{pca}-freq{nbmots}.pdf')
    # faire l'histogramme de la longueur des phrases en mots:
    longueurs = [len(i) for i in phrasesLemmatisees]
    
    plt.figure()
    plt.hist(longueurs, bins=40)
    plt.xlabel('Longueur en mots')
    plt.ylabel('Nombre de phrases')
