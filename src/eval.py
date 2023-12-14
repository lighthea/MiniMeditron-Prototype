import os
import sys
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

logger = logging.getLogger(__name__)

def main():
    # Load configuration
    conf_file = sys.argv[1]
    config = load_config(conf_file)
    config["general_settings"]["task"] = "eval" # Override the task to evaluation
    config["dataset_generation"]["with_token"] = True

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
    else:
        model = AutoModelForCausalLM.from_pretrained(config["general_settings"]['base_model_id'],
                                                     quantization_config=bnb_config,
                                                     torch_dtype=torch.float32,
                                                     use_flash_attention_2=config["general_settings"]["use_flash_attn"],
                                                     device_map={"": Accelerator().process_index})

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["general_settings"]['base_model_id'], add_eos_token=True)
        tokenizer.pad_token = tokenizer.eos_token
    
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

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        query_tensors = batch["input_ids"]

        for query in query_tensors:
            outputs = model(torch.reshape(query, (1,-1,)))

        pass

    return


if __name__ == '__main__':
    main()
