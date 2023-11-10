import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os.path as p
import os

def create_tf_idf(data_folder, vectorizer):
    guidelines = []
    #for folder in os.listdir(data_folder):
     #   for filename in os.listdir(p.join(data_folder, folder)):
    #x = os.walk(data_folder)
    files = [p.join(x[0],file_path) for x in os.walk(data_folder) for file_path in x[2]]
    for file in files:
        with open(file, 'r') as f:
            guidelines.append(f.read().lower().replace('\n', ' ').replace('\r', ''))

    # Initialize a TF-IDF vectorizer and fit it to the guidelines
    return vectorizer.fit_transform(guidelines)

def retrieve_top_k_guidelines(query, tf_idf_matrix, vectorizer, data_folder, k=5):
    # Vectorize the query
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity between the query and all guidelines
    cosine_similarities = linear_kernel(query_vector, tf_idf_matrix).flatten()

    # Get top k indices of guidelines that are most similar to the query
    top_indices = cosine_similarities.argsort()[-k:][::-1]
    return return_guidelines(top_indices, data_folder)

def return_guidelines(top_indices, data_folder):
    guidelines = [p.join(x[0],file_path) for x in os.walk(data_folder) for file_path in x[2]]
    top_guidelines = [guidelines[index] for index in top_indices]
    for guide in top_guidelines:
        with open(guide, 'r') as f:
            yield f.read()

def create_matrix(tf_idf_path):
    if not p.isfile(p.join(tf_idf_path, "tfidf.pkl")):
        vectorizer = TfidfVectorizer(stop_words='english')
        tf_idf_matrix = create_tf_idf(data_folder, vectorizer)
        with open(p.join(tf_idf_path, "tfidf.pkl"), 'wb') as file:
            pickle.dump(tf_idf_matrix, file)
        with open(p.join(tf_idf_path,'tfidf_vectorizer.pkl'), 'wb') as file:  
            pickle.dump(vectorizer, file)
    else:
        with open(p.join(tf_idf_path, "tfidf.pkl"), 'rb') as file:
            tf_idf_matrix= pickle.load(file)
        with open(p.join(tf_idf_path,'tfidf_vectorizer.pkl'), 'rb') as file:  
            vectorizer = pickle.load(file)
    return tf_idf_matrix, vectorizer