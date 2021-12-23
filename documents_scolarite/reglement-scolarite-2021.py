# -*- coding: utf-8 -*-



from io import StringIO
import re as regex


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage



def remove_non_ascii(text):
    return "\n".join([word for word in text if ord(word) < 128])

def load_data(max_pages=100):
    retsrt = StringIO()
    device = TextConverter(PDFResourceManager(), retsrt, laparams=LAParams(), codec = 'utf-8')
    interpreter = PDFPageInterpreter(PDFResourceManager(), device=device)
    filepath = open('./reglement-scolarite-2021.pdf', 'rb')
    
    for page in PDFPage.get_pages(filepath, set(), maxpages=max_pages, caching=True, check_extractable=True):
        interpreter.process_page(page)
        text_data = retsrt.getvalue()
    filepath.close()
    device.close()
    retsrt.close()
    return text_data


def enregistrer_texte(fichier):
    with open(fichier, 'w', encoding='utf-8') as f:
        f.write(texte)
    f.close()



whitelist = '\x01\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !"#$%&\'’()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~«»ôûâîàéêçèùÉÀÈÇÂÊÎÔÛÄËÏÖÜÀÆæÇÉÈŒœÙﬁ—'


texte = load_data()

texte = regex.sub(rf'[^{whitelist}]', "", texte) # ne garder que les caractères de la whitelist


texte = texte.replace("-\n", "")         # raccorder les mots coupés

texte = texte.replace("\x0cRÈGLEMENT DE SCOLARITÉ DE L’ECOLE CENTRALE DE LYON\n\n2021-2022\n\n","").replace("\x0c","") # supprimer les entêtes (headers)
texte = regex.sub("[0-9]+\n{2}",u'',texte) # supprimer la numérotation en pide de page (footers)



texte = regex.sub("\([^()]*\)",u'',texte)     # enlever les parenthèses et leur contenu
# permet de transformer "l’(les) année(s) universitaires" en "l' année universitaires"
texte = texte.replace("’", "'")    # changer les types d'apostrophes
texte = texte.replace("l ?", "l'") # supprimer les bugs de formatage du document
texte = texte.replace("' ", "'")     # enlever les espaces après les apostrophes
texte = regex.sub(" +",u' ',texte)   # supprimer les répétitions d'espaces
texte = regex.sub(" +([,.])",u'\1',texte)  # supprimer les espaces avant une virgule ou un point
texte = texte.replace("ﬁ", "fi") # remplacement des ligatures
texte = texte.replace("æ", "ae") 
texte = texte.replace("œ", "oe")
texte = texte.replace(u"U+0001", "")


 # à ce stade on a déjà un document assez épuré et très lisible


texte = regex.sub(r"(.+)\n{1}", r"\1 ", texte) # supprimer les retours à la ligne dans un paragraphe (i.e. sauts de ligne doubles)





texte = regex.sub(r'([;:.]) *\n—', r' ', texte) # supprimer les retours à la ligne de listes à puces

texte = regex.sub(r'—', ' ', texte)



texte = regex.sub(r"(Article [0-9]+ – )", r"\n\n", texte) # rajouter des sauts de ligne entre les articles et virer les titres


texte = regex.sub(r"[\n][A-Z]{1}\.{1}[0-9]{1}[^\.\n,]{1}.+\n{1}", "\n\n", texte) # supprimer les titres de paragraphes d'annexes tupe "L.1 Principe \n"


texte = regex.sub(r"[A-Z]{1} {1}.+\n{1}", "\n\n", texte) # supprimer les titres de paragraphes d'annexes tupe "L Mobilité à l'international \n"




'''

texte = regex.sub(r"[A-Z]+\.[0-9]\.{1}[0-9]\.{1}[0-9](.+\n{1})", r"\1\n", texte) # supprimer les titres de type B.1.2.3
texte = regex.sub(r"[A-Z]+\.[0-9]\.{1}[0-9](.+\n{1})", r"\1\n", texte) # supprimer les titres de type B.1.2
texte = regex.sub(r"[A-Z]+\.[0-9](.+\n{1})", r"\1\n", texte) # supprimer les titres de type B.1


'''

enregistrer_texte('./reglement-scolarite-2021.txt')


    
    
#paragraphes = ''.join(e for e in paragraphes is e.isalnum())


paragraphes = texte.split("\n")

