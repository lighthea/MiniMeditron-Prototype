import os
import sys
import torch
from random import random

from torch.optim import Adam
from accelerate import Accelerator
from peft import LoraConfig
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer
from trl.core import LengthSampler

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.training import load_config, init_wandb_project, init_configs
from lib.dataset import load_dataset
from tqdm import tqdm

# Data collators are objects that will form a batch by using a list of dataset elements 
# as input. These elements are of the same type as the elements of `train_dataset` or 
# `eval_dataset`.
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def main():
    # Load configuration
    conf_file = sys.argv[1]
    config = load_config(conf_file)
    
    model_name = config["general_settings"]["base_model_id"]
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=config["model_parameters"]["learning_rate"],
        # gradient_accumulation_steps=config["model_parameters"]["gradient_accumulation_steps"],
        # optimize_cuda_cache=True,
        # log_with="wandb",
        batch_size=config["model_parameters"]["per_device_train_batch_size"],
        # mini_batch_size=1,
        # task_name=config["wandb_parameters"]["run_name"],
        # tracker_project_name=config["wandb_parameters"]["wandb_project"],
    )

    # Initialize the wandb project
    init_wandb_project(config)

    # Initialize the accelerator and quantization configs
    # Not used in practice (I have no clue on how to make it work with PPO trainer)
    bnb_config, ia3_config = init_configs(config)

    # Now let's build the model, the reference model, and the tokenizer. We first
    # load the model in bfloat16 to save memory using `transformers`
    model = AutoModelForCausalLM.from_pretrained(ppo_config.model_name, torch_dtype=torch.bfloat16)

    # And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        device_map={ "": Accelerator().local_process_index }
    )

    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name) #, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'

    # Define the reward function as a random number between 0 and 1
    def compute_rewards(texts):
        return torch.rand(len(texts))

    dataset: DatasetDict = load_dataset(config, tokenizer)
    # dataset = dataset.rename_column("text", "query")

    input_size = LengthSampler(2, 512)

    def tokenize(sample):
        prompt = sample["text"]

        sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    dataset = dataset.map(tokenize, batched=False)
    dataset.remove_columns("text")
    dataset.set_format('torch')

    # We make sure to use the `Adam` optimizer on the model parameters that requires gradients
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=ppo_config.learning_rate)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        dataset=dataset["train"],
        data_collator=collator,
        optimizer=optimizer,
        tokenizer=tokenizer,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        # "eos_token_id": tokenizer.eos_token_id,
        # "max_new_tokens": 512
    }

    # tokenizer.padding_side = 'left'

    # output_min_length = 
    # output_max_length = 512
    output_length_sampler = LengthSampler(4, 128)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from SFTModel
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        print('>>> egouyv <<<')

        # Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = compute_rewards(texts)
        rewards = [torch.tensor(output) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        print(stats)

if __name__ == "__main__":
    main()
