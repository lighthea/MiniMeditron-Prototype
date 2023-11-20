import os
import sys

import torch
from transformers import AutoTokenizer
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.training import load_config, load_dataset, init_wandb_project


def main():
    # Load configuration
    conf_file = sys.argv[1]
    config = load_config(conf_file)
    model_name = config["model_parameters"]["base_model_id"]
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1.41e-5,
        log_with="wandb"
    )

    # Initialize the wandb project
    init_wandb_project(config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    # Define the reward function as a random number between 0 and 1
    def reward_model(x):
        return torch.rand(10)

    train_dataset = load_dataset(config, tokenizer, None, with_context=True, with_token=False, with_output=False)
    train_dataset = train_dataset.rename_column("text", "query")

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
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
    }

    from tqdm import tqdm
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        print(stats)


if __name__ == "__main__":
    main()
