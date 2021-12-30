# -*- coding: utf-8 -*-



import gensim

import nltk

import sklearn
import re as regex
import string
import spacy


# création des ensembles de stopwords
stopwordsNLTK = set(nltk.corpus.stopwords.words('french'))
from spacy.lang.fr.stop_words import STOP_WORDS as stopwordsSpacy


# librement inspiré du tutpo Gensim sur Kaggle
class WordEmbedding():
    def __init__(self, chemin):
        self.text = None
        self.path = chemin
        self.dimension = None
        self.win_size = None
        self.mode = None
        

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
                    if token.ent_type_ != '':
                        explodedSentence.append(token.text.lower()) if token.text.lower() not in stopwordsSpacy else None
                    else:
                        explodedSentence.append(token.lemma_.lower()) if token.text.lower() not in stopwordsSpacy else None
            lemmatizedSentencesSpacy.append(explodedSentence)
        return lemmatizedSentencesSpacy
        
   
    def word2vec(self,phrases,dimension=250,taille_fenetre=6,mode=1):
        '''phrases doit être une liste de liste de strings dont chacun est un mot
           mode 1 pour skip gram, 0 pour cbow'''
        self.dimension = dimension
        self.win_size = taille_fenetre
        self.mode = 'Skip-Gram' if mode == 1 else 'CBOW'
        return gensim.models.Word2Vec(sentences=phrases, window=taille_fenetre, min_count=1, sg=mode, vector_size=dimension) 
    
    
    def visualiserMotsFrequents(self,model,nbmax):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
         
        import seaborn as sns
        sns.set_style("darkgrid")
        

        whitelist = regex.compile('[^ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzôûâîàéêçèùÉÀÈÇÂÊÎÔÛÄËÏÖÜÀÇÉÈÙ]')
        mots_selectionnes = [word for word in model.wv.index_to_key[:nbmax] if whitelist.search(word) is None]
        vecteurs = np.empty((len(mots_selectionnes), self.dimension), dtype='f')
        
        for i in range(len(mots_selectionnes)):
            vecteurs[i] = model.wv[mots_selectionnes[i]]
        
        
        color_list  = len(mots_selectionnes) * ['blue']
        
        
        # Trouver les coordonnées de  t-SNE en deux dimensions
        np.set_printoptions(suppress=True)        
        # Réduire la domention de 300 à 10 avec PCA
        pca_components = 10
        reduc = sklearn.decomposition.PCA(n_components=pca_components).fit_transform(vecteurs)
        
        
        
        Y = sklearn.manifold.TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
        
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
                         scatter_kws={'s': 30,
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
                    ).set_size(15)
        
        
        plt.xlim(Y[:, 0].min()*1.05, Y[:, 0].max()*1.05)
        plt.ylim(Y[:, 1].min()*1.05, Y[:, 1].max()*1.05)
                
        plt.title(f'Visualisation des {nbmax} mots les plus fréquents\nDimension départ {self.dimension} +PCA -> {pca_components} + t-SNE -> 2\nSkip-gram fenêtre {self.win_size}', size='medium')
        
    
    
    
    
if __name__ == "__main__":
    path = './données_scolarité_acquises/'
    wr = WordEmbedding(path)
    wr.importerCorpus('corpus.txt')
    sentences = wr.segmenterTexteNLTK()
    wr.definirModeleSpacy('fr_core_news_sm')
    phrases = wr.segmenterTexteNLTK()
    phrasesTokeniseesSpacy = wr.tokeniserPhrasesSpacy(phrases)
    wr.enregistrerPhrases(phrasesTokeniseesSpacy,'phrases.txt')
    phrasesLemmatiseesSpacy = wr.lemmatiserPhrasesSpacy(phrasesTokeniseesSpacy)
    wr.enregistrerPhrases(phrasesLemmatiseesSpacy,'phrases-lemmatisées.txt')
    skipgram = wr.word2vec(phrasesLemmatiseesSpacy,dimension=250,taille_fenetre=6,mode=1)
    wr.visualiserMotsFrequents(skipgram,50)
    
    
