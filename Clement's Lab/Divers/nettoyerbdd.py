# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:12:07 2021

@author: cleme
"""

data_path = "fra.txt"

input_characters = set()

fichier = ""

caractere_interdit = ['!', '"', '$', '%', '&', "'", '(', ')', '+', '\xa0', '«', '»', '\u2009', '\u200b', '‘', '’', '…', '\u202f', '‽', '₂', ':', ';','\xad','–', '—', '€','º','/']

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

with open("PhrasesFr.txt","w") as ff:
    
    for line in lines:
            try:
                _, phrase, _ = line.split("\t")
                
                #phrase = phrase.replace(".","")
                ecrire = True
                '''
                for c in phrase:
                    if c in caractere_interdit:
                        ecrire = False
                        break
                '''
                if ecrire:
                    ff.write(phrase+ "\n")
                    for char in phrase:
                        if not char in input_characters:
                            input_characters.add(char)
            except:
                pass
            
            finally:
                f.close()

print(sorted(list(input_characters)))





