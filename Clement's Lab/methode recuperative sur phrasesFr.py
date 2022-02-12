import numpy as np
from tensorflow import keras
import gensim
import nltk
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as stopwordsSpacy


num_samples = 1000


#récupération du modèle
model = keras.models.load_model("surPhrasesFr")
encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

#le modele de word2vec
nlp = spacy.load('fr_core_news_sm')
skipgram = gensim.models.Word2Vec.load("wvsurPhrasesFr.model")







def Seq2Vec(s):
    s_tokenisee = nlp(s)
    explodedSentence = []
    for token in s_tokenisee:
        if not token.is_punct:
            if token.ent_type_ != '': # déterminer si le token fait partie d'une entité
                explodedSentence.append(token.text.lower()) if token.text.lower() not in stopwordsSpacy else None
            else:
                explodedSentence.append(token.lemma_.lower()) if token.text.lower() not in stopwordsSpacy else None
    
    s_lematisee = explodedSentence
    
    liste_vecteurs = np.array([[skipgram.wv[u] for u in s_lematisee]])
    if liste_vecteurs.size != 0:#parfois après lemmatisation,la phrase est vide, le code suivant évite les crashs mais ne résout pas le pb
        VecteurPensee_temp = encoder_model.predict(liste_vecteurs)
        VecteurPensee = np.concatenate((VecteurPensee_temp[0][0],VecteurPensee_temp[1][0]))# La LSTM contient deux états internes. On les concatene
        
        return(VecteurPensee)
    else:
        print("phrase vide après lemmatisation")
        return(10*np.ones(1024))
    
 
    
#génération des vecteurs des questions
path = 'PhrasesFr.txt'

    
with open(path, 'r', encoding='utf-8') as f:
   texte = f.read()

phrases = nltk.tokenize.sent_tokenize(texte, language='french')[:num_samples]

CouplesVecteursQuestions = [(Seq2Vec(s),s) for s in phrases]

def PhraseLaPlusProche(s):
    vec = Seq2Vec(s)
    rep = ""
    dist = np.linalg.norm(vec -  CouplesVecteursQuestions[0][0])
    for c in CouplesVecteursQuestions:
        dist_temp =  np.linalg.norm(vec - c[0])
        if dist_temp <= dist:
            rep = c[1]
            dist = dist_temp
    return(rep)
        

 
    
    