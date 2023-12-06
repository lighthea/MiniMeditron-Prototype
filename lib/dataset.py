import json
import os
import random

from datasets import Dataset, DatasetDict
from tqdm import tqdm

from lib.tf_idf import batch_bm25
from lib.utils import retrieve_prompt
from lib.tf_idf import retrieve_n_best_guidelines

import pandas as pd
import json

def blanket(config: dict) -> str:
    file = config["general_settings"]['process_file']
    with open(file, 'r') as f:
        data = json.load(f)
    # Checks if document structure is not an empty dict
    if not data["document_structure"]:
        return "LABEL"
    data["document_structure"]["Condition"] = "LABEL"
    return json.dumps(data["document_structure"])


def load_pretrained_dataset(config: dict) -> DatasetDict | None:
    """
    Load the tokenized dataset if it exists and if the user doesn't want to re-tokenize it
    :param config:  The configuration file
    :return: The tokenized dataset if it exists, None otherwise
    """
    # Check if the dataset has already been tokenized
    print("Checking if tokenized dataset exists")
    if (os.path.exists(config["model_folders"]['tokenized_data_path']) and
            not config["dataset_generation"]['force_retokenize']):
        # check if directory is not empty
        if os.listdir(config["model_folders"]['tokenized_data_path']):
            print("Loading tokenized dataset")
            # Load the tokenized dataset
            dataset = DatasetDict.load_from_disk(config["model_folders"]['tokenized_data_path'])
            return dataset

    return None


def construct_raw_dataset(config: dict) -> Dataset:
    """
    Construct the raw dataset from the jsonl files
    :param config:  The configuration file
    :return: The raw dataset
    """
    # For each patient, retrieve the top k guidelines
    queries = []
    labels = []
    for file in tqdm(os.listdir(config["general_folders"]['train_folder']), desc="Loading dataset"):
        if file.endswith(".jsonl"):
            with open(os.path.join(config["general_folders"]['train_folder'], file)) as f:
                for line in f:
                    data = json.loads(line)
                    queries.append(data["structure"])
                    labels.append(data["label"])

    dataset = Dataset.from_dict({"text": queries, "labels": labels})
    return dataset


def add_bm25_context(dataset: Dataset, config: dict) -> Dataset:
    """
    Add the top k guidelines to the dataset
    :param dataset: The dataset
    :param config:  The configuration file
    :return: The dataset with the top k guidelines
    """
    dataset = batch_bm25(dataset, config["general_folders"]['guidelines_folder'],
                         n=config["dataset_generation"]['n_context_guidelines'],
                         base_folder=config["general_folders"]['base_folder'])
    return dataset


def fill_prompt_with_dataset(config: dict, dataset: Dataset) -> Dataset:
    """
    Fill the prompt with the dataset
    :param config: the configuration file
    :param dataset: the dataset
    :return: a query column with the prompt filled with the dataset
    """

    def create_query(partial_prompt_, x, with_context_):
        """
        Create the query from the partial prompt and the dataset
        :param partial_prompt_: the uncompleted prompt with the INPUT and CONTEXT placeholders
        :param x: the dataset row
        :param with_context_: whether the prompt contains the CONTEXT placeholder
        :return: a query with the prompt filled with the dataset
        """
        blanket_string = blanket(config).replace("LABEL", x["labels"])
        query = partial_prompt_.replace("INPUT", str(x["text"]))
        return {"query": query.replace("CONTEXT", str(x["context"])) if with_context_ else query,
                "labels": blanket_string}

    # Format the labels
    partial_prompt = retrieve_prompt(config["general_settings"]['process_file'])
    with_context = "CONTEXT" in partial_prompt

    if with_context:
        dataset = add_bm25_context(dataset, config)

    remove_columns = ["text"]
    if with_context:
        remove_columns.append("context")
    return dataset.map(lambda x: create_query(partial_prompt, x, with_context),
                       remove_columns=remove_columns)


def format_chat_for_qa(example, config, tokenizer):
    """
    Format the chat for the QA task with or without the output
    :param example: the data row
    :param config: the configuration file
    :param tokenizer: the tokenizer
    :return: a chat formatted for the QA task
    """
    chat_template = [{"role": "user",
                      "content": example["query"]}]
    if config["dataset_generation"]["with_output"]:
        chat_template.append({"role": "assistant",
                              "content": example["labels"]})

    tokenized_output = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=False)
    return {"text": tokenized_output}


def format_chat_for_preference_optimisation(example, dataset: Dataset, tokenizer):
    # select a wrong label
    wrong_label = random.choice([label for label in dataset["labels"] if label != example['labels']])
    chat_template_wrong = [{"role": "assistant",
                            "content": wrong_label}]
    chat_template_right = [{"role": "assistant",
                            "content": example["labels"]}]
    tokenized_output_wrong = tokenizer.apply_chat_template(chat_template_wrong, tokenize=False,
                                                           add_generation_prompt=False)
    tokenized_output_right = tokenizer.apply_chat_template(chat_template_right, tokenize=False,
                                                           add_generation_prompt=False)

    return {"rejected": tokenized_output_wrong, "chosen": tokenized_output_right}


def save_dataset(dataset: Dataset, config: dict):
    """
    Save the dataset to the disk
    :param dataset: the dataset
    :param config: the configuration file
    """
    dataset = dataset.shuffle()
    dataset = dataset.train_test_split(test_size=config["dataset_generation"]["test_size"], shuffle=True)

    # Create the folder if it doesn't exist
    if not os.path.exists(config["model_folders"]['tokenized_data_path']):
        os.makedirs(config["model_folders"]['tokenized_data_path'])

    # Save the tokenized dataset
    dataset.save_to_disk(config["model_folders"]['tokenized_data_path'])
    return dataset


def generate_contrastive_dataset(config, dataset, n_pairs=10000, n_retrieval=10):
    pd_dataset = pd.DataFrame(dataset)
    pd_dataset["labels"] = pd_dataset["labels"].apply(lambda label : label.lower())

    for _ in range(n_pairs):
        guideline = pd_dataset.sample()

        # On average should be far from anchor (hopefully)
        neg_pair = pd_dataset.sample()

        pos_pair = NotImplemented()


    # groupped = pd_dataset.groupby("labels")
    # return groupped

def load_dataset(config: dict, tokenizer) -> DatasetDict:
    """
    Load the dataset from the disk or create it if it doesn't exist
    :param config: the configuration file
    :param tokenizer: the tokenizer
    :return: the dataset
    """

    Load the tokenized dataset if it exists
    dataset = load_pretrained_dataset(config)
    if dataset is not None:
        return dataset

    # Construct the raw dataset
    dataset = construct_raw_dataset(config)

    if config["general_settings"]["task"] == "cl":
        return generate_contrastive_dataset(config, dataset)

    # Fill the prompt with the dataset
    dataset = fill_prompt_with_dataset(config, dataset)

    # If the model is QA based, format the chat for the QA task
    dataset = dataset.map(lambda x: format_chat_for_qa(x, config, tokenizer),
                          remove_columns=["query"])

    # If the model is preference based, format the chat for the preference optimisation task
    if config["general_settings"]["task"] == "po":
        dataset = dataset.map(lambda x: format_chat_for_preference_optimisation(x, dataset, tokenizer))
        dataset = dataset.rename_column("text", "prompt")
    dataset = dataset.remove_columns(["labels"])

    def tokenize(example):
        if config["dataset_generation"].get("padding_side") is not None:
            tokenizer.padding_side = config["dataset_generation"]["padding_side"]
        tokenized = tokenizer.encode(example["text"], add_special_tokens=True,
                                     padding="max_length")
        return {"input_ids": tokenized}

    # Tokenize the dataset
    if config["dataset_generation"]["with_token"]:
        dataset = dataset.map(tokenize)

    # Save the dataset
    dataset = save_dataset(dataset, config)

    return dataset
