import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest

from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def preprocess(doc):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(doc.lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    tokens = " ".join(tokens)
    return tokens

dictionary_documents = []

documents = []
folder_path = "../Guidelines/split_guidelines/wikidoc.jsonl"
for file in os.listdir(folder_path):
    if file.endswith(".json"):
        with open(f"{folder_path}/{file}", 'r') as f:
            data = json.load(f)
            preprocess_doc = preprocess(data["text"])
            documents.append(preprocess_doc)
            dictionary_documents.append({data["title"]: preprocess_doc})

# vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(documents)

# dim red
n_components = 100  # You can adjust this number based on your dataset size and available computational power
svd = TruncatedSVD(n_components=n_components)
X_reduced = svd.fit_transform(X)

clf = IsolationForest(contamination=0.15)  # contamination parameter can be tuned
preds = clf.fit_predict(X_reduced)

anomalies = np.where(preds == -1)[0]
anomalous_docs = [documents[i] for i in anomalies]

print("Anomalous Documents:")
for doc in anomalous_docs:
    for dictionary in dictionary_documents:
        if doc == dictionary[list(dictionary.keys())[0]]:
            print(list(dictionary.keys())[0])
            break

print("Length of Anomalous Documents:", len(anomalous_docs))

with open("non_disease_guidelines_names/anomalous_titles_015.txt", "w") as f:
    for doc in anomalous_docs:
        for dictionary in dictionary_documents:
            if doc == dictionary[list(dictionary.keys())[0]]:
                f.write(list(dictionary.keys())[0] + "\n")
                break