import json
import os
import re
import sys

from datasets import Dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from lib.utils import yield_structured_obj, repair_json


def base_word_corpus(files):
    data = ""
    for structure_file in files:
        with open(structure_file, 'r') as f:
            data = data + "\n" + json.load(f)
    return re.findall(r'\b[a-zA-Z0-9]+\b', data)


def preprocess_json_for_bm25(json_obj: str, base_corpus: list[str]):
    # Load the JSON string into a Python dictionary

    data = json_obj
    # Initialize an empty list to store the cleaned words
    current_corpus = re.findall(r'\b[a-zA-Z0-9]+\b', data)
    if base_corpus is None:
        return current_corpus
    return [word for word in current_corpus if word not in base_corpus]


def process_folder_for_bm25(folder_path: str, base_corpus: list[str]) -> list[str]:
    # Loop through each file in the folder
    for json_obj in yield_structured_obj(folder_path):
        yield preprocess_json_for_bm25(str(json_obj), base_corpus=base_corpus)


def init_bm25(corpus_folder_path: str, base_file_folder: str) -> (BM25Okapi, list[str]):
    # Recursively get all the files in the folder
    base_files = []
    for root, dirs, files in os.walk(base_file_folder):
        for file in files:
            if file.endswith(".json"):
                base_files.append(os.path.join(root, file))
    base_corpus = base_word_corpus(base_files)
    return BM25Okapi(list(process_folder_for_bm25(corpus_folder_path, base_corpus))), base_corpus


def retrieve_n_best_guidelines(query: str, bm25: BM25Okapi, guidelines: list[str], base_corpus: list[str], n: int = 3):
    # Loads the JSON string into a Python dictionary
    # Retrieve the top n guidelines
    top_n_guidelines = bm25.get_top_n(preprocess_json_for_bm25(query, base_corpus), guidelines, n=n)
    # Return the top n guidelines
    return "\n\n".join(top_n_guidelines)


def batch_bm25(dataset: Dataset, guideline_folder: str, base_folder: str, n: int = 3):
    print("Initializing BM25 model")
    bm25, base_corpus = init_bm25(guideline_folder, base_folder)
    guidelines = list(map(lambda x: str(x), yield_structured_obj(guideline_folder)))
    dataset = dataset.map(
        lambda x: {"context": retrieve_n_best_guidelines(x["text"], bm25, guidelines, n=n, base_corpus=base_corpus)})
    filtered_dataset = dataset.filter(lambda x: x["context"] is not None)

    print(f"Filtered {(len(dataset) - len(filtered_dataset) / len(dataset)) * 100}% of the dataset")
    return filtered_dataset
