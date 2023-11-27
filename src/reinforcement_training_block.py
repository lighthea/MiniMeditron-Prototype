import os
import sys
from random import random
from secure_env import secure_config

from accelerate import Accelerator
from peft import LoraConfig
from datasets import DatasetDict
from transformers import AutoTokenizer
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.training import load_config, init_wandb_project, init_configs
from lib.dataset import load_dataset
from tqdm import tqdm


def main():
    # Load configuration
    conf_file = sys.argv[1]
    config = load_config(conf_file)
    config = secure_config(config)
    
    model_name = config["general_settings"]["base_model_id"]
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=config["model_parameters"]["learning_rate"],
        gradient_accumulation_steps=config["model_parameters"]["gradient_accumulation_steps"],
        optimize_cuda_cache=True,
        log_with="wandb",
        batch_size=config["model_parameters"]["per_device_train_batch_size"],
        mini_batch_size=1,
        task_name=config["wandb_parameters"]["run_name"],
        tracker_project_name=config["wandb_parameters"]["wandb_project"],
    )

    # Initialize the wandb project
    init_wandb_project(config)

    # Initialize the accelerator and quantization configs
    # Not used in practice (I have no clue on how to make it work with PPO trainer)
    bnb_config, ia3_config = init_configs(config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        peft_config=ia3_config,
        load_in_8bit=True,
        device_map={ "": Accelerator().local_process_index }
    )

    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Define the reward function as a random number between 0 and 1
    def compute_rewards(query, response):
        print(query)
        print(response)
        return random()

    train_dataset: DatasetDict = load_dataset(config, tokenizer)
    train_dataset = train_dataset.rename_column("text", "query")

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"], padding="max_length", max_length=4096)
        return sample

    train_dataset = train_dataset.map(tokenize, batched=False)
    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        dataset=train_dataset["train"],
        tokenizer=tokenizer,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 1024
    }

    tokenizer.padding_side = 'left'

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        print(batch)
        query_tensors = batch["input_ids"]

        # Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Compute reward score
        rewards = [compute_rewards(q, r) for q, r in zip(batch["query"], batch["response"])]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, [response_tensors], rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        print(stats)

if __name__ == "__main__":
    main()
