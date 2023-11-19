import json
import os
import re

import numpy as np
import wandb
from tqdm import tqdm


def decode_predictions(tokenizer, predictions):
    labs = np.where(predictions.label_ids != -100, predictions.label_ids, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    preds = np.where(predictions.predictions != -100, predictions.predictions, tokenizer.pad_token_id)
    prediction_text = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return {"labels": labels, "predictions": prediction_text}


def yield_structured_obj(folder):
    for file in sorted(os.listdir(folder)):

        if file.endswith(".jsonl"):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(str(line))
                        yield data["structure"]

                    except json.JSONDecodeError:
                        # Handle the case where the input is not a valid JSON string
                        print(line)
                        continue


def replace_string_in_files(folder_path, old_string, new_string):
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".jsonl"):  # Ensuring only structured files are processed
            file_path = os.path.join(folder_path, filename)

            # Read the contents of the file
            with open(file_path, 'r') as file:
                filedata = file.read()

            # Replace the target string
            filedata = filedata.replace(old_string, new_string)

            # Write the file out again
            with open(file_path, 'w') as file:
                file.write(filedata)


def retrieve_prompt(config: dict) -> str:
    file = config['process_file']
    with open(file, 'r') as f:
        data = json.load(f)
    return data["prompt"].replace("OUTPUT", str(data["document_structure"]))


def repair_json(json_string):
    # Pattern to identify key-value pairs where value is unquoted
    pattern = r'(?<=:)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(,|\})'

    # Function to replace unquoted values with quoted ones
    def replace_with_quotes(match):
        value = match.group(1)
        # Avoid quoting true, false, null, or numbers
        if value.strip() in ["true", "false", "null"] or re.match(r'^-?\d+(\.\d+)?$', value):
            return f" {value}{match.group(2)}"
        return f' "{value}"{match.group(2)}'

    # Use regular expression to replace unquoted values
    json_string = re.sub(pattern, replace_with_quotes, json_string)
    repaired_json_string = re.sub(r',\s*([\]}])', r'\1', json_string)
    return (repaired_json_string.replace("'s", "s")
            .replace("'", '"'))

