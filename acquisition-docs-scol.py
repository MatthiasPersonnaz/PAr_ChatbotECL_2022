import string
import re as regex



class TraitementDocuments():
    
    def __init__(self, chemin, liste_docs, nom_fichier_compil):
        self.path = chemin
        self.compilDocName = nom_fichier_compil
        self.docNamesList = liste_docs
        self.whitelist = '-\n\',-.[]()0123456789:;ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzôûâîàéêçèùÉÀÈÇÂÊÎÔÛÄËÏÖÜÀÆæÇÉÈŒœÙﬁﬂ— '
        self.alphanumlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzôûâîàéêçèùÉÀÈÇÂÊÎÔÛÄËÏÖÜÀÇÉÈÙ'
    
    
    
    
    def importerDocument(self):
        texte = ''
        for i in range(len(self.docNamesList)):
            f = open(self.path + self.docNamesList[i], 'r', encoding="utf-8")
            texte += f.read()
            f.close()
        f = open(self.compilDocName, 'w', encoding="utf-8")
        f.write(texte)
        f.close()
        
    
    
    def nettoyerDoc(self):
        with open(self.compilDocName, 'r', encoding="utf-8") as f:
            texte = f.read()
        
        f = open(self.compilDocName, 'w', encoding="utf-8")
        f.truncate(0)
        texte = self.homogeneiserCaracteres(texte)
        texte = self.supprimerMauvaisFormatage(texte)
        f.write(texte)

    

    
    def homogeneiserCaracteres(self, texte):
        texte = texte.replace("ﬁ", "fi") # remplacement des ligatures
        texte = texte.replace("æ", "ae") 
        texte = texte.replace("œ", "oe")
        texte = texte.replace("ﬂ", "fl")
        texte = texte.replace("’", "'")    # changer les types d'apostrophes
        texte = texte.replace("…", ".")
        texte = texte.replace('—', '-') # changer les tirets longs
        texte = texte.replace('...', '.')
        texte = texte.replace('..', '.')
        texte = regex.sub(rf'[^{self.whitelist}]', "", texte) # ne garder que les caractères autorisés dans ce qui reste
        return texte        
        
        
    def supprimerMauvaisFormatage(self, texte):
        texte = regex.sub(r"\([^()]*\)",'',texte)     # enlever les parenthèses et leur contenu
        texte = texte.replace(r"[ldj] ?", "l'") # supprimer les espaces après une apostrophe
        texte = regex.sub(r"' +", "'", texte)     # enlever les espaces après les apostrophes
        texte = regex.sub(r" +([,\.])",r'\1',texte)  # supprimer les espaces avant une virgule ou un point
        texte = regex.sub(r' +\.', r'\.', texte) # supprimer les espaces avant un point final
        texte = texte.replace(r"-\n", "")         # raccorder les mots coupés dans un paragraphe
        texte = regex.sub(r" +",' ',texte)   # supprimer les répétitions d'espaces
        return texte


    def supprimerMiseEnPagePubliPDF(self, texte):
        texte = texte.replace(r"\x0cRÈGLEMENT DE SCOLARITÉ DE L’ECOLE CENTRALE DE LYON\n\n2021-2022\n\n","").replace("\x01","") # supprimer les entêtes (headers)        
        texte = regex.sub(r"(.+)\n{1}", r"\1 ", texte) # supprimer les retours à la ligne dans un paragraphe (i.e. sauts de ligne doubles)
        texte = regex.sub(r'([;:.]) *\n—', r' ', texte) # supprimer les retours à la ligne de listes à puces
        texte = regex.sub(r"(Article [0-9]+ – )", r"\n\n", texte) # rajouter des sauts de ligne entre les articles et virer les titres
        texte = regex.sub(r"[\n][A-Z]{1}\.{1}[0-9]{1}[^\.\n,]{1}.+\n{1}", "\n\n", texte) # supprimer les titres de paragraphes d'annexes type "L.1 Principe \n"
        texte = regex.sub(rf"[A-Z]{1} {1}[{self.alphanumlist}]+\n{1}", "\n\n", texte) # supprimer les titres de paragraphes d'annexes du  type "L Mobilité à l'international \n"
        texte = regex.sub(r"[A-Z]+\.[0-9]\.{1}[0-9]\.{1}[0-9](.+\n{1})", r"\1\n", texte) # supprimer les titres de type B.1.2.3
        
        texte = regex.sub(r"[A-Z]+\.[0-9]\.{1}[0-9](.+\n{1})", r"\1\n", texte) # supprimer les titres de type B.1.2
        texte = regex.sub(r"[A-Z]+\.[0-9](.+\n{1})", r"\1\n", texte) # supprimer les titres de type B.1
        texte = regex.sub("[0-9]+\n{2}",u'',texte) # supprimer la numérotation en pide de page (footers)
        return texte
    
    


        
   
        
if __name__ == '__main__':
    liste_documents = ['reglement-scolarite-2021_clean.txt', 'consignes-mobilite-covid19.txt', 'dispense-de-mobilite-obligatoire.txt', 'mobilite-3a-explication-de-la-procedure-intranet.txt', 'mobilite-cesure-choix-post-2a.txt', 'procedure-selection-ceu.txt', 'sejour-international.txt', 'validation-post-tfe.txt', 'double-cursus-l3-maths.txt']
    
    t = TraitementDocuments('./documents_scolarite/', liste_documents, 'corpus.txt')
    t.importerDocument()
    t.nettoyerDoc()



