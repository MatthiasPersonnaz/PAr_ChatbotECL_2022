# PAr102-chatbot

GitLab du PAr 102 de l'année 2021-2022 visant à créer un Chatbot ou assistant conversationnel pour renseigner sur le programme de formation de l'école Centrale de Lyon

Vous trouverez dans le Clement's Lab un ensemble de codes permettant la réalisation d'une méthode récupérative.
Le code méthode récupérative sur phrasesFr code une fonction calculant la phrase "la plus proche" de l'utilisateur parmis les 10000 premières phrases du fichier phrasesFR. Ce code ne renvoie pas de réponse à la question mais il est possible de l'adapter en quelques minutes en se munnisant d'une bdd de questions réponses.
La notion de plus proche phrase est au sens des moindres carrés pour la version vectorisée de la hrase via le modèle de ML.
Les modèles de word2vec et d'encodage sont créés dans les fichiers Auto-Encodeur LSTM sur {...}. Il faut enregistre rles modèles à la main à l'aide des méthodes .save des différentes librairies.
 


À la base, vous trouverez:
* dans le fichier `word2vec.py` le code pour fabriquer le Word2Vec sur les documents de scolarité (se trouvant dans le dossier `documents_scolarité`), dont les résultats des calculs (modèle Gensim, phrases lemmatisées et extraites, fichiers binaires des vecteurs et vocabulaire) se trouvent dans `word2vec_docs_scol_traités`.
* dans le fichier `autoencodeur.py` un code d'autoencodeur classique sans résultats probant (avec des couches denses);
* dans le fichier `autoencodeur2.py` un code d'autoencodeur simple à base de LSTM, sans pouvoir récupérer les vecteurs de contexte;
* dans le fichier `autoencodeur3.py` un code d'autoencodeur permettant de récupérer les vecteurs de contexte;
* dans le fichier `acquisition-docs-scol` des méthodes qui ont servi à extraire le texte des documents PDF de la scolarité, ces codes ne servent plus et les acquisitions de texte ont été finalisées et nettoyées à la main, donc ne surtout pas exécuter à nouveau son contenu.

# Installer TensorFlow
Ci-dessous, la méthode qui a servi à installer TensorFlow avec les versions compativles de cudnn

```
conda install -c conda-forge python=3.9.9 # tf supporte pas 3.10
conda install -c conda-forge spyder
pip install tensorflow-gpu # il a mis la 2.7.0
pip install tensorflow # ??? (update non inutile)
conda install -c conda-forge cudnn # et a mis la 8.2.1.32 et cudatoolkit 11.5.0 en dependency
```
Correspondance des compatibilités de TensorFlow avec CUDA et cuDNN: https://www.tensorflow.org/install/source#gpu

Correspondance des compatibilités de TensorFlow avec le paquet `tensorflow-addons`: https://github.com/tensorflow/addons#python-op-compatility

