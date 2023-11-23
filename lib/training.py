import json
import os

import torch
import wandb
from datasets import Dataset, DatasetDict
from peft import IA3Config, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, DPOTrainer
from lib.tf_idf import batch_bm25
from lib.utils import retrieve_prompt
from lib.wandb import retrieve_checkpoint


def init_configs():
    bf16_support = torch.cuda.is_bf16_supported()
    float_type = torch.bfloat16 if bf16_support else torch.float16
    print(f"Using {float_type} for training")
    print("Initializing accelerator and quantization configs")

    # Initialize the quantization config

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # Initialize the IA3 config
    ia3_config = IA3Config(
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        feedforward_modules=["down_proj"]
    )

    return bnb_config, ia3_config


def setup_model_and_training_finetuning(config: dict, bnb_config: BitsAndBytesConfig, ia3_config: IA3Config):
    # Initialize the accelerator and quantization configs
    if config["wandb_parameters"]["start_from_checkpoint"]:
        folder = retrieve_checkpoint(config)
        model = AutoModelForCausalLM.from_pretrained(folder,
                                                     quantization_config=bnb_config,
                                                     use_flash_attention_2=True
                                                     )
        tokenizer = AutoTokenizer.from_pretrained(folder, add_eos_token=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(config["general_settings"]['base_model_id'],
                                                     quantization_config=bnb_config,
                                                     use_flash_attention_2=True
                                                     )

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["general_settings"]['base_model_id'], add_eos_token=True)
        tokenizer.pad_token = tokenizer.eos_token

    # Set up model for training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, ia3_config)

    print({"trainable_params": model.print_trainable_parameters()})
    train_args = TrainingArguments(
        output_dir=config["model_folders"]['output_dir'],
        warmup_steps=5,
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        save_strategy="steps",  # Save the model checkpoint every logging step
        evaluation_strategy="steps",  # Evaluate the model every logging step
        do_eval=True,
        report_to=["wandb"],
        run_name=config["wandb_parameters"]["run_name"],
        load_best_model_at_end=True,
        bf16=torch.cuda.is_bf16_supported(),
        bf16_full_eval=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        fp16_full_eval=not torch.cuda.is_bf16_supported(),
        **config["model_parameters"]
    )

    return model, tokenizer, train_args


def create_all_path(config: dict):
    for key in config["model_folders"].keys():
        if not os.path.exists(config["model_folders"][key]):
            os.makedirs(config["model_folders"][key])

    for key in config["general_folders"].keys():
        if not os.path.exists(config["general_folders"][key]):
            os.makedirs(config["general_folders"][key])


def load_config(config_file: str) -> dict:
    with open(config_file) as config_file:
        config = json.load(config_file)

    create_all_path(config)
    return config


def init_wandb_project(config: dict) -> None:
    # Wandb Login
    print("Logging into wandb")
    wandb.login(key=config["wandb_parameters"]['wandb_key'])
    if len(config["wandb_parameters"]["wandb_project"]) > 0:
        os.environ["WANDB_PROJECT"] = config["wandb_parameters"]["wandb_project"]
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def launch_training(model, tokenizer, train_args, dataset, ia3_conf, config):
    if config["general_settings"]["task"] == "qa":
        return launch_training_qa(model, tokenizer, train_args, dataset, ia3_conf)
    elif config["general_settings"]["task"] == "finetune":
        return launch_training_finetune(model, tokenizer, train_args, dataset, ia3_conf)
    elif config["general_settings"]["task"] == "po":
        return launch_training_po(model, tokenizer, train_args, dataset, ia3_conf)


def launch_training_finetune(model, tokenizer, train_args, dataset, ia3_conf):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=ia3_conf,
        dataset_text_field="text",
        dataset_batch_size=10,
    )

    return trainer


def launch_training_po(model, tokenizer, train_args, dataset, ia3_conf):
    max_seq_length = max(len(tokenizer.encode(example["prompt"])) for example in
                         tqdm(dataset["train"], desc="Estimating max prompt length"))
    print(f"Max prompt length: {max_seq_length}")
    max_target_length = max(len(tokenizer.encode(example["chosen"])) for example in
                            tqdm(dataset["train"], desc="Estimating max target length"))
    print(f"Max target length: {max_target_length}")
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=ia3_conf,
        loss_type="ipo",
        beta=0.1,
        max_length=max_seq_length,
        max_prompt_length=max_seq_length,
        max_target_length=max_target_length,
        padding_value=tokenizer.pad_token_id,
        label_pad_token_id=tokenizer.pad_token_id,
        truncation_mode="keep_end",
    )

    return trainer


def launch_training_qa(model, tokenizer, train_args, dataset, ia3_conf):
    instruction_template = "<|user|>"
    response_template = "<|assistant|>"
    # Checks if the instruction template is the first token of the first prompt
    instruction_template_ids = tokenizer.encode(instruction_template, add_special_tokens=False)
    response_template_ids = tokenizer.encode("\n" + response_template, add_special_tokens=False)[2:]

    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template_ids,
                                               response_template=response_template_ids,
                                               tokenizer=tokenizer,
                                               mlm=False)

    # Determine max seq length
    max_seq_length = max(len(tokenizer.encode(example["text"])) for example in tqdm(dataset["train"],
                                                                                    desc="Estimating max seq length"))
    print(f"Max seq length: {max_seq_length}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=ia3_conf,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=max_seq_length + 1,
        dataset_batch_size=10,
    )

    return trainer
