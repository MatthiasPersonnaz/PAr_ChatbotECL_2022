# -*- coding: utf-8 -*-

import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from io import StringIO

import pdfminer
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
    filepath = open('./documents_scolarite/reglement-scolarite-2021.pdf', 'rb')
    
    for page in PDFPage.get_pages(filepath, set(), maxpages=max_pages, caching=True, check_extractable=True):
        interpreter.process_page(page)
        text_data = retsrt.getvalue()
    filepath.close()
    device.close()
    retsrt.close()
    return text_data





texte = load_data()
paragraphes = texte.split("\n")
#paragraphes = ''.join(e for e in paragraphes is e.isalnum())

