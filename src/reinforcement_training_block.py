import os
import sys

import torch
from datasets import DatasetDict
from transformers import AutoTokenizer
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer
from accelerate import Accelerator


from secure_env import *

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.training import load_config, init_configs
from lib.wandb import init_wandb_project
from tqdm import tqdm
from lib.dataset import load_dataset


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def main():
    # Load configuration
    conf_file = sys.argv[1]
    
    config = load_config(conf_file)
    # config = secure_config(config)

    model_name = config["general_settings"]["base_model_id"]
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1.41e-5,
        batch_size=256,
        optimize_device_cache=True,
        # gradient_accumulation_steps=config["model_parameters"]["gradient_accumulation_steps"]
        # log_with="wandb"
    )

    # Initialize the wandb project
    # init_wandb_project(config)

    bnb_config, ia3_conf = init_configs(config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name, device_map={ "": Accelerator().local_process_index })
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Define the reward function as a random number between 0 and 1
    def reward_model(queries, responses):
        # print(x)
        return torch.rand(10)

    train_dataset: DatasetDict = load_dataset(config, tokenizer)
    train_dataset = train_dataset.rename_column("text", "query")

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"], padding="max_length", max_length=2048)
        sample["query"] = tokenizer.decode(sample["input_ids"])

        return sample

    train_dataset = train_dataset.map(tokenize, batched=False)
    train_dataset.set_format("torch")

    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        dataset=train_dataset["train"],
        tokenizer=tokenizer,
        data_collator=collator
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 512
    }


    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        print(f">>> Epoch {epoch} <<<", end='\n')
        
        query_tensors = batch["input_ids"]
        print(len(query_tensors))

        # Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        print("[x] - Got tensors from trainer", end='\n')
        
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        print("[x] - Tokenized", end='\n')

        # Compute reward score
        # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(batch["query"], batch["response"])
        print("[x] - Computed Reward", end='\n')
        
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, [response_tensors], rewards)
        print("[x] - Step done", end='\n')
        
        ppo_trainer.log_stats(stats, batch, rewards)
        print(stats)


if __name__ == "__main__":
    main()
