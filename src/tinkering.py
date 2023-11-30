import re
import json
import numpy as np
from os.path import join
from os import listdir
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def build_tfidf(guidelines: list[str]):
    # Preprocess all documents (prior to building the vocabulary)
    corpus = []

    print(' - Proprocess and extract the corpus')
    for guideline in tqdm(guidelines):
        text = guideline["text"]

        # Replacing unconventional punctuation with spaces
        text = re.sub(r'[\(\)\[\]\,\-\;\.\!\?”“\"\']', ' ', text)
        text = re.sub(r'[\*+\_\#]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Stemming is done by sklearn
        corpus.append(text)
    
    # Build the vocabulary matrix
    print(' - Vectorize the corpus')
    vectorizer = CountVectorizer(min_df=10, max_df=0.7) # Must be less that 80%
    counts = vectorizer.fit_transform(corpus).toarray()

    # Learn the IDF matrix
    print(' - Learn the IDF for each terms')
    transformer = TfidfTransformer(smooth_idf=True)
    transformer.fit(counts)

    # Build the TF-IDF matrix for each sample of the corpus
    print(' - Build the TF-IDF dense matrix')
    tfidf = transformer.transform(counts + 1).toarray() # Smoothing of the count

    # Running PCA on tf-idf matrix to select the important words
    pca = PCA(n_components=100)
    pca.fit(tfidf)

    # Finally transform the tfidf matrix into a smaller collection
    compact_tfidf = pca.transform(tfidf)
    return compact_tfidf, pca, transformer

GUIDELINE_PATH = join('.', 'Guidelines', 'meditron-guidelines', 'processed')

# Retrieve all guidelines
print(' - Retrieving guidelines')
guidelines = []
for file in listdir(GUIDELINE_PATH):
    path = join(GUIDELINE_PATH, file)

    if path.endswith('.jsonl'):
        with open(path, 'r') as f:
            guidelines += [x for x in list(map(json.loads, f.readlines())) if 'title' in x]
guidelines = guidelines[:5000]

# Preprocess all texture (prior to building the vocabulary)
compact_idf, pca, transformer = build_tfidf(guidelines)

# Transform pca invert
X = np.zeros(shape=100)
X[0] = 1

Y = pca.inverse_transform(X)

breakpoint()

