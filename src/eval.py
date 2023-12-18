import os
import sys
import json
import logging

from tqdm import tqdm
from accelerate import Accelerator
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from lib.training import init_configs, setup_model_and_training_finetuning, load_config, init_wandb_project
from lib.dataset import load_dataset
from lib.utils import repair_json

logger = logging.getLogger(__name__)

def main():
    # Load configuration
    conf_file = sys.argv[1]
    config = load_config(conf_file)
    config["general_settings"]["task"] = "eval" # Override the task to evaluation
    config["dataset_generation"]["with_token"] = True
    config["keep_labels"] = True

    # Initialize the wandb project
    init_wandb_project(config)

    # Initialize the accelerator and quantization configs
    bnb_config, ia3_conf = init_configs(config)

    # Set up the model for inference
    if config["wandb_parameters"]["start_from_checkpoint"]:
        model = AutoModelForCausalLM.from_pretrained(config["chekpoint_folder"],
                                                     quantization_config=bnb_config,
                                                     torch_dtype=torch.float32,
                                                     use_flash_attention_2=config["general_settings"]["use_flash_attn"],
                                                     device_map={"": Accelerator().process_index})
        tokenizer = AutoTokenizer.from_pretrained(config["chekpoint_folder"], add_eos_token=True)
        tokenizer.padding_side = 'left'
    else:
        model = AutoModelForCausalLM.from_pretrained(config["general_settings"]['base_model_id'],
                                                     quantization_config=bnb_config,
                                                     torch_dtype=torch.float32,
                                                     use_flash_attention_2=config["general_settings"]["use_flash_attn"],
                                                     device_map={"": Accelerator().process_index})

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["general_settings"]['base_model_id'], add_eos_token=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    
    # Load the dataset
    dataset = load_dataset(config, tokenizer)
    dataset.set_format('torch')

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    eval_dataset = dataset["test"]
    eval_sampler = SequentialSampler(eval_dataset)
    # batch_size = config["model_parameters"]["per_device_eval_batch_size"]

    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,
        collate_fn=collator,
    )

    # Eval
    logger.info("***** Running evaluation {} *****".format(config["wandb_parameters"]["baseline_name"]))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", 1)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    kwags = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 1024,
    }

    invalid_json = 0
    total_correct_prediction = 0
    total_prediction = 0
    errors = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        query = batch["input_ids"][0]
        labels = batch["labels"][0]

        output = model.generate(torch.reshape(query, (1,-1,)), **kwags)

        result = tokenizer.decode(output[0])

        # Retrieve the result of the model (by cutting on the tag <|assistant|>)
        string = result.split('<|assistant|>')[-1].split('<|system|>')[0].replace('</s>', '').replace('<s>', '').replace('\n', '')

        # Repair json
        # I know, this is a very bad practice and may break but oh look a butterfly 🦋
        j = repair_json(string)
        l = repair_json(labels)

        try:
            j = json.loads(j)
            l = json.loads(l)
        except ValueError:
            invalid_json += 1
            continue

        j = json.dumps(j)
        l = json.dumps(l)

        # Retrieve the condition
        if j == l:
            total_correct_prediction += 1
        else:
            errors.append((j, l))

        total_prediction += 1

    print("Accuracy: {:.2f} %".format(100 * total_correct_prediction / (total_prediction + invalid_json)))
    print("Invalid JSONs: {} (accounting for {:.2f} %)".format(invalid_json, 100.0 * total_prediction / (total_prediction + invalid_json)))

    print("{} errors: ".format(len(errors)))
    for i in range(min(10, len(errors))):
        print(" - Predicting {} instead of {}".format(*errors[i]))

    return


if __name__ == '__main__':
    main()
