# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 18:25:47 2021

@author: matth
"""




# <scolarité>
# <évaluation-connaissances-compétences>
# <césure-et-mobilité>
# <scolarité-hors-ecl>
# <jurys>
# <commission-évolution-règlement-scolarité>

import nltk
import re

with open('./corpus-balisé-sousech.txt', 'r', encoding='utf-8') as f:
    texte = f.read()

extendedAlphaNum = '\'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzôûâîàéêçèùÉÀÈÇÂÊÎÔÛÄËÏÖÜÀÇÉÈÙ-'

balises = set(re.findall(rf'<([{extendedAlphaNum}]+)>', texte))
# paragraphes = re.findall(rf'<[{extendedAlphaNum}]+>([^<>]+)', texte)



for bb in balises:
    with open(f'./catégories/{bb}.txt', 'w', encoding='utf-8') as f:
        for paragraphe in re.findall(rf"<{bb}>([^<>]+)", texte):
            f.write(paragraphe)


# créer un fichier par phrase
for bb in balises:
    paragraphe = '\n'.join(re.findall(rf"<{bb}>([^<>]+)", texte)) # texte de la catégorie
    p = nltk.tokenize.sent_tokenize(paragraphe, language='french') # liste des phrases de chaque catégorie
    for i in range(len(p)):
        with open(f'./catégories/sousech/{bb}{i}.txt', 'w', encoding='utf-8') as f:
            f.write(p[i])
                
            
