import json
import os
import sys

from datasets import Dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from lib.utils import yield_structured_obj, repair_json


def preprocess_json_for_bm25(json_obj: dict):
    # Load the JSON string into a Python dictionary

    data = json_obj
    # Initialize an empty list to store the cleaned words
    cleaned_words = []

    # Iterate over key-value pairs
    for key, value in data.items():
        # Consider both keys and values, ensure they are strings
        key_value_str = str(key) + " " + str(value)

        # Convert to lowercase
        key_value_str = key_value_str.lower()

        # Split into words and extend the list
        cleaned_words.extend(key_value_str.split())

        return cleaned_words


def process_folder_for_bm25(folder_path: str) -> list[str]:
    # Loop through each file in the folder
    for json_obj in yield_structured_obj(folder_path):
        yield preprocess_json_for_bm25(json_obj)


def init_bm25(corpus_folder_path: str) -> BM25Okapi:
    return BM25Okapi(list(process_folder_for_bm25(corpus_folder_path)))


def retrieve_n_best_guidelines(query: str, bm25: BM25Okapi, guidelines: list[str], n: int = 3):
    # Loads the JSON string into a Python dictionary
    query = repair_json(str(query))
    try:
        data = json.loads(query)
    except json.JSONDecodeError as e:
        # Handle the case where the input is not a valid JSON string
        print(query)
        return None
    # Retrieve the top n guidelines
    top_n_guidelines = bm25.get_top_n(preprocess_json_for_bm25(data), guidelines, n=n)
    # Return the top n guidelines
    return "\n\n".join(top_n_guidelines)


def batch_bm25(dataset: Dataset, guideline_folder: str, n: int = 3):
    print("Initializing BM25 model")
    bm25 = init_bm25(guideline_folder)
    guidelines = list(map(lambda x: str(x), yield_structured_obj(guideline_folder)))
    dataset = dataset.map(lambda x: {"context": retrieve_n_best_guidelines(x["text"], bm25, guidelines, n=n)})
    filtered_dataset = dataset.filter(lambda x: x["context"] is not None)

    print(f"Filtered {len(dataset) - len(filtered_dataset)/len(dataset)*100}% of the dataset")
    return filtered_dataset
