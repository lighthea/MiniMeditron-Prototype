from sentence_transformers import SentenceTransformer, util

import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.dataset import Dataset

model = SentenceTransformer('all-MiniLM-L6-v2')


def insert_semantic_embeddings(dataset: Dataset, col_name="embeddings"):
    return dataset.map(lambda x : {
        col_name : model.encode(x["labels"])
        })


def cos_sim_based_pref(example, choice1, choice2, tokenizer):
    score1 = util.cos_sim(example["embeddings"], choice1["embeddings"])
    score2 = util.cos_sim(example["embeddings"], choice2["embeddings"])

    chosen = choice1 if score1 > score2 else choice2 
    rejected = choice2 if score1 > score2 else choice1

    chat_template_wrong = [{"role": "assistant",
                            "content": rejected["labels"]}]
    chat_template_right = [{"role": "assistant",
                            "content": chosen["labels"]}]

    tokenized_output_wrong = tokenizer.apply_chat_template(chat_template_wrong, tokenize=False,
                                                           add_generation_prompt=False)
    tokenized_output_right = tokenizer.apply_chat_template(chat_template_right, tokenize=False,
                                                           add_generation_prompt=False)

    return {"text" : example["text"], "rejected": tokenized_output_wrong, "chosen": tokenized_output_right}


def cos_sim_based_pair(example, dataset: Dataset, tokenizer):
    iterable_dataset = dataset.to_iterable_dataset()

    min_sim = float('inf')
    max_sim = 0
    correct_label = None
    wrong_label = None

    for x in iterable_dataset:
        score = util.cos_sim(x["embeddings"], example["embeddings"])
        if max_sim < score:
            max_sim = score
            correct_label = x["labels"]
        
        if min_sim > score:
            min_sim = score
            wrong_label = x["labels"]

        #  = dataset.max(lambda x : ))["labels"]
        # wrong_label = dataset.min(lambda x : util.cos_sim(x["embeddings"], example["embeddings"]))["labels"]

    chat_template_wrong = [{"role": "assistant",
                            "content": wrong_label}]
    chat_template_right = [{"role": "assistant",
                            "content": correct_label}]
    tokenized_output_wrong = tokenizer.apply_chat_template(chat_template_wrong, tokenize=False,
                                                           add_generation_prompt=False)
    tokenized_output_right = tokenizer.apply_chat_template(chat_template_right, tokenize=False,
                                                           add_generation_prompt=False)

    return {"rejected": tokenized_output_wrong, "chosen": tokenized_output_right}