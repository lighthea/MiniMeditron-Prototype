import json
import os
import sys
import torch
from trl import SFTTrainer

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.wandb import init_wandb_project
from lib.training import init_configs, setup_model_and_training, blanket, load_dataset, create_all_path


def main():
    # Load configuration
    conf_file = sys.argv[1]
    with open(conf_file) as config_file:
        config = json.load(config_file)

    # Create all paths
    create_all_path(config)

    # Initialize the wandb project
    init_wandb_project(config)

    # Initialize the accelerator and quantization configs
    bnb_config, ia3_conf = init_configs(torch.cuda.is_bf16_supported())

    # Set up model for training
    model, tokenizer, train_args = setup_model_and_training(config, bnb_config, ia3_conf)

    # Load the dataset
    dataset = load_dataset(config, tokenizer, blanket(config))

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=ia3_conf,
        dataset_text_field="text",
    )

    trainer.train()


if __name__ == "__main__":
    main()
