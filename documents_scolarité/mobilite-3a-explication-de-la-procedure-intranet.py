# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:18:21 2021

@author: matth
"""


from io import StringIO
import re as regex



f = open('./mobilite-3a-explication-de-la-procedure-intranet_raw.txt', 'r', encoding='utf-8')
texte = f.read()
f.close()



whitelist = '!"#$%&\'’()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~«»ôûâîàéêçèùÉÀÈÇÂÊÎÔÛÄËÏÖÜÀÆæÇÉÈŒœÙﬁ—\n '




texte = regex.sub(rf'[^{whitelist}]', "", texte)






texte = texte.replace("-\n", " ")         # raccorder les mots coupés

texte = regex.sub(r'(- )+', ' ', texte)

texte = regex.sub("\([^()]*\)",u'',texte)     # enlever les parenthèses et leur contenu


texte = regex.sub(r"\n+", r" ", texte) # supprimer les retours à la ligne dans un paragraphe

texte = regex.sub(r" +", " ", texte) # supprimer les espaces en trop

with open('./mobilite-3a-explication-de-la-procedure-intranet.txt', 'w', encoding='utf-8') as f:
    f.write(texte)
f.close()


