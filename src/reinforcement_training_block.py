import os
import sys

import torch
from datasets import DatasetDict
from transformers import AutoTokenizer
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer

from secure_env import *

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.training import load_config, init_wandb_project, init_configs
from tqdm import tqdm
from lib.dataset import load_dataset


def main():
    # Load configuration
    conf_file = sys.argv[1]
    
    config = load_config(conf_file)
    config = secure_config(config)

    model_name = config["general_settings"]["base_model_id"]
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1.41e-5,
        log_with="wandb"
    )

    # Initialize the wandb project
    init_wandb_project(config)

    bnb_config, ia3_conf = init_configs(config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Define the reward function as a random number between 0 and 1
    def reward_model(x):
        print(x)
        return torch.rand(10)

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
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 1024
    }

    tokenizer.padding_side = 'left'

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        print(f">>> Epoch {epoch} <<<")
        
        query_tensors = batch["input_ids"]

        # Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        print("[x] - Got tensors from trainer")
        
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        print("[x] - Tokenized")

        # Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        print("[x] - Computed Reward")
        
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, [response_tensors], rewards)
        print("[x] - Step done")
        
        ppo_trainer.log_stats(stats, batch, rewards)
        print(stats)


if __name__ == "__main__":
    main()
