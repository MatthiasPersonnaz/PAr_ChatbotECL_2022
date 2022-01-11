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
    filepath = open('./dispense-de-mobilite-obligatoire.pdf', 'rb')
    
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



whitelist = '!"#$%&\'’()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~«»ôûâîàéêçèùÉÀÈÇÂÊÎÔÛÄËÏÖÜÀÆæÇÉÈŒœÙﬁ—\n '


texte = load_data()

texte = regex.sub(rf'[^{whitelist}]', "", texte)






texte = texte.replace("-\n", " ")         # raccorder les mots coupés

texte = regex.sub(r'(- )+', ' ', texte)

texte = regex.sub("\([^()]*\)",u'',texte)     # enlever les parenthèses et leur contenu


texte = regex.sub(r"\n+", r" ", texte) # supprimer les retours à la ligne dans un paragraphe

texte = regex.sub(r" +", " ", texte) # supprimer les espaces en trop




enregistrer_texte('./dispense-de-mobilite-obligatoire.txt')



