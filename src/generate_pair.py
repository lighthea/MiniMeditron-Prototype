def format_pair(tokenizer, wrong_label, right_label):
    chat_template_wrong = [{"role": "assistant",
                            "content": wrong_label}]
    chat_template_right = [{"role": "assistant",
                            "content": right_label]
    tokenized_output_wrong = tokenizer.apply_chat_template(chat_template_wrong, tokenize=False,
                                                           add_generation_prompt=False)
    tokenized_output_right = tokenizer.apply_chat_template(chat_template_right, tokenize=False,
                                                           add_generation_prompt=False)

    return {"rejected": tokenized_output_wrong, "chosen": tokenized_output_right}

def label_vs_random(datapoint, dataset: Dataset, tokenizer):
    # select a wrong label
    wrong_label = random.choice([label for label in dataset["labels"] if label != example['labels']])
    right_label = examples["labels"]

    return format_pair(tokenizer, wrong_label, right_label)
