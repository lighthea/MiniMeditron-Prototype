import matplotlib.pyplot as plt
import gensim
from gensim import corpora
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import os
import json
# nltk.download('stopwords')
# nltk.download('punkt')


documents = []
folder_path = "../Guidelines/split_guidelines/wikidoc.jsonl"
for file in os.listdir(folder_path):
    if file.endswith(".json"):
        with open(f"{folder_path}/{file}", 'r') as f:
            data = json.load(f)
            documents.append(data["title"])

def preprocess(doc):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(doc.lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return tokens

tokenized_docs = [preprocess(doc) for doc in documents]

dictionary = corpora.Dictionary(tokenized_docs)

corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

num_topics = 5
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

for i in range(num_topics):
    terms = [t[0] for t in lda_model.show_topic(i, 5)]
    print(f"Topic {i + 1}: {', '.join(terms)}")

def plot_top_words(lda_model, n_words=5):
    topics = lda_model.show_topics(num_topics=num_topics, num_words=n_words, formatted=False)
    
    fig, axes = plt.subplots(1, num_topics, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    
    for idx, topic in topics:
        topic_words = dict(topic)
        wordcloud = WordCloud(width=400, height=400, background_color='white', random_state=1).generate_from_frequencies(topic_words)
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(f'Topic {idx + 1}', fontdict=dict(size=16))
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

plot_top_words(lda_model)
