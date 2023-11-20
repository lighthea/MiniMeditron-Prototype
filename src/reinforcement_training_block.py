import sys

from transformers import AutoTokenizer
from trl import PPOConfig, AutoModelForCausalLMWithValueHead

from lib.training import load_config


def main():
    # Load configuration
    conf_file = sys.argv[1]
    config = load_config(conf_file)

    ppo_config = PPOConfig(
        model_name=config["model_parameters"]["base_model_id"],
        learning_rate=1.41e-5,
        log_with="wandb"
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    tokenizer.pad_token = tokenizer.eos_token
