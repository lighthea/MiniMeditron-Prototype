import os
import sys
import torch
from random import random

from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.training import load_config, init_configs
from lib.dataset import load_dataset
from tqdm import tqdm

# Create the ppo config
config_path = sys.argv[1]
config = load_config(config_path)

# Retrieve the model name
model_name = config["general_settings"]["base_model_id"]

# Build the ppo config
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=config["model_parameters"]["learning_rate"],
    ppo_epochs=100,
    mini_batch_size=1,
    batch_size=4,
    optimize_device_cache=True,
    gradient_accumulation_steps=1
)

# GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# Build the dataset and tokenize the sample(s)
dataset = load_dataset(config, tokenizer)
# dataset = dataset.rename_column("text", "query")

input_size = LengthSampler(2, 512)

def tokenize(sample):
    prompt = sample["text"]

    sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

dataset = dataset.map(tokenize, batched=False)
dataset.remove_columns('text')
dataset.set_format('torch')

dataset = dataset["train"]

# Set the seed for initializing value head for deterministic eval
set_seed(42)

# Now let's build the model, the reference model, and the tokenizer. We first
# load the model in bfloat16 to save memory using `transformers`
model = AutoModelForCausalLM.from_pretrained(ppo_config.model_name, torch_dtype=torch.bfloat16)

# And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

# We create a reference model by sharing 20 layers
ref_model = create_reference_model(model, num_shared_layers=20)

# We make sure to use the `Adam` optimizer on the model parameters that requires gradients
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=ppo_config.learning_rate)

# What the f*ck is a collator
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# We then build the PPO Trainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    data_collator=collator,
    dataset=dataset,
    optimizer=optimizer,
)

# We then build the reward pipeline, we will use the random model to compute the reward
def compute_rewards(texts):
    return torch.rand(len(texts))

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_length_sampler = LengthSampler(4, 128)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from the policy model
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    print('PBUzoyefbrosuygrouyvg')

    # Compute rewards using our own reward function
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = compute_rewards(texts)
    rewards = [torch.tensor(output) for output in pipe_outputs]

    # Run a PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    print(stats)

    # Save model every 100 epochs
    # if epoch % 100 == 0:
    #     if ppo_trainer.accelerator.is_main_process:
    #         ppo_trainer.save_pretrained(model_save_path)
