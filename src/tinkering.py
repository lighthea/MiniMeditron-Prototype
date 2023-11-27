import re
import json
import numpy as np
from os.path import join
from os import listdir
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.cluster import DBSCAN

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
print(' - Preprocessing text')
stemmer =  SnowballStemmer('english')
corpus = []
for guideline in tqdm(guidelines):
    text = guideline["text"]

    # Replacing unconventional punctuation with spaces
    text = re.sub(r'[\(\)\[\]\,\-\;\.\!\?”“\"\']', ' ', text)
    text = re.sub(r'[\*+\_\#]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Stemming (done by sklearn)
    # text = stemmer.stem(text)
    corpus.append(text)

# Build the vocabulary matrix
print(' - Vectorize the corpus')
vectorizer = CountVectorizer(min_df=10)
counts = vectorizer.fit_transform(corpus).toarray()


# Learn the IDF matrix
print(' - Learn the IDF matrix')
tftransformer = TfidfTransformer()
tftransformer.fit(counts)

# Build the TF-IDF matrix for each sample of the corpus
print(' - Finally build the TF-IDF matrix for each sample of the corpus')
tf_idfs = tftransformer.transform(counts + 1).toarray() # We cheat a little bit with the count

def kullback_leibler_divergence(P, Q):
    return np.dot(P, np.log(P / Q))

# Clustering of condition based on the tf_idfs matrix
# Currently assume equal distribution of each disease(s)
def distance(c1, c2):
    M = 0.5 * c1 + 0.5 * c2 # Mixture distribution
    return 1/2 * kullback_leibler_divergence(c1, M) + 1/2 * kullback_leibler_divergence(c2, M)

# Build the distance matrix (may take a while)

