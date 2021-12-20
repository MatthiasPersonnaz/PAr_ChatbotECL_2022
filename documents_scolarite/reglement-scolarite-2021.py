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



whitelist = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !"#$%&\'’()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~«»ôûâîàéêçèùÉÀÈÇÂÊÎÔÛÄËÏÖÜÀÆæÇÉÈŒœÙﬁ—'


texte = load_data()

texte = regex.sub(rf'[^{whitelist}]', "", texte)


texte = texte.replace("-\n", "")         # raccorder les mots coupés

texte = texte.replace("\x0cRÈGLEMENT DE SCOLARITÉ DE L’ECOLE CENTRALE DE LYON\n\n2021-2022\n\n","").replace("\x0c","") # supprimer les entêtes
texte = regex.sub("[0-9]+\n{2}",u'',texte) # supprimer les numéros de pages en footers





texte = regex.sub("\([^()]*\)",u'',texte)     # enlever les parenthèses et leur contenu
# permet de transformer "l’(les) année(s) universitaires" en "l' année universitaires"
texte = texte.replace("’", "'")
texte = texte.replace("l ?", "l'") # supprimer les bugs de formatage du document
texte = texte.replace("' ", "'") # enlever les espaces après les apostrophes
texte = texte.replace("ﬁ", "fi") # remplacement des ligatures
texte = texte.replace("æ", "ae") 
texte = texte.replace("œ", "oe")


texte = regex.sub(r"(.+)\n{1}", r"\1 ", texte) # supprimer les retours à la ligne dans un paragraphe



texte = regex.sub(r"(Article [0-9]+ – )", r"\n\n", texte) # rajouter des sauts de ligne entre les paragraphes
# texte = regex.sub(r"([A-Z]+\.{1}[0-9]+.+)\n{1}", r"\n\n", texte) # rajouter les sauts de ligne entre les paragraphes annexes et supprimer les titres de paragraphes d'annexe


texte = regex.sub(r"([A-Z]+\.{1}[0-9])\n{1}", r"", texte)


enregistrer_texte('./reglement-scolarite-2021.txt')


    
    
#paragraphes = ''.join(e for e in paragraphes is e.isalnum())


paragraphes = texte.split("\n")

