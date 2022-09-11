# PAr102-chatbot

GitLab du PAr 102 de l'année 2021-2022 visant à créer un Chatbot ou assistant conversationnel pour renseigner sur le programme de formation de l'école Centrale de Lyon



À la base, vous trouverez:
- dans le fichier `word2vec.py` le code pour fabriquer le Word2Vec sur les documents de scolarité (se trouvant dans le dossier `documents_scolarité`), dont les résultats des calculs (modèle Gensim, phrases lemmatisées et extraites, fichiers binaires des vecteurs et vocabulaire) se trouvent dans `word2vec_docs_scol_traités`. Si les dépendances sont satisfaites (Spacy, NLTK, sklearn, pandas, seaborn, gensim), une exécution s'occupe automatiquement de créer un word2vec des documents présents dans le git;
- dans le fichier `graphes-étude-sémantique.py` un code pour étudier la distribution sémantique des Word2Vec réalisés;
- dans le fichier `questions_scolarité.xlsx` l'ensemble d'entraînement pour le modèle QAnet;
- dans le fichier `question_answering_QAnet_création_dataset.ipynb` l'ensemble des codes pour créer le dataset de QAnet à partir de l'excel précédemment évoqué, ce code enregistre le dataset au format binaire avec la librairie `pickle` dans le fichier `questions_réponses_scol`;
- dans le fichier `question_answering_QAnet_entraînement.ipynb` le modèle QAnet.

# Installer TensorFlow
Ci-dessous, la méthode qui a servi à installer TensorFlow avec les versions compatibles de cudnn

```
conda install -c conda-forge python=3.9.9 # tf supporte pas 3.10
conda install -c conda-forge spyder
pip install tensorflow-gpu # il a mis la 2.7.0
pip install tensorflow # ??? (update non inutile)
conda install -c conda-forge cudnn # et a mis la 8.2.1.32 et cudatoolkit 11.5.0 en dependency
```
Correspondance des compatibilités de TensorFlow avec CUDA et cuDNN: https://www.tensorflow.org/install/source#gpu

Correspondance des compatibilités de TensorFlow avec le paquet `tensorflow-addons`: https://github.com/tensorflow/addons#python-op-compatility

