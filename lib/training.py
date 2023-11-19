import json
import os

import torch
from datasets import Dataset
from peft import IA3Config, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from lib.tf_idf import batch_bm25
from lib.utils import retrieve_prompt


def blanket(config: dict) -> str:
    file = config['process_file']
    with open(file, 'r') as f:
        data = json.load(f)

    data["document_structure"]["Condition"] = "LABEL"
    return json.dumps(data["document_structure"])


def init_configs(bf16_support: bool):
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


def load_dataset(config: dict, tokenizer, blanket_string: str = None) -> [dict]:
    # Check if the dataset has already been tokenized
    print("Checking if tokenized dataset exists")
    if os.path.exists(config['tokenized_data_path']) and not config['force_retokenize']:
        print("Loading tokenized dataset")
        # Load the tokenized dataset
        dataset = Dataset.load_from_disk(config['tokenized_data_path'])
        return dataset

    # For each patient, retrieve the top k guidelines
    queries = []
    labels = []
    for file in tqdm(os.listdir(config['train_folder']), desc="Loading dataset"):
        if file.endswith(".jsonl"):
            with open(os.path.join(config['train_folder'], file)) as f:
                for line in f:
                    data = json.loads(line)
                    queries.append(data["structure"])
                    labels.append(data["condition_name"])

    dataset = Dataset.from_dict({"text": queries, "labels": labels})
    del queries, labels

    # Append the guidelines to the dataset
    dataset = batch_bm25(dataset, config['guidelines_folder'], n=config['n_context_guidelines'],
                         base_folder=config['base_folder'])

    # Merge the query and context into a single string using the prompt defined in the structure file
    partial_prompt = retrieve_prompt(config)
    dataset = dataset.map(lambda x: {"query": partial_prompt
                          .replace("INPUT", str(x["text"]))
                          .replace("CONTEXT", str(x["context"]))}
                          , remove_columns=["text", "context"])

    # Tokenize the dataset
    def transform_example(example):
        assistant_prompt = ""
        if blanket_string is None:
            assistant_prompt = example["labels"]
        else:
            assistant_prompt = blanket_string.replace("LABEL", example["labels"])

        tokenized_output = {"text": tokenizer.apply_chat_template([
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": assistant_prompt}
        ], tokenize=False, padding="max_length", add_generation_prompt=False)}
        # Convert tensor output to a dictionary format suitable for the dataset
        return tokenized_output

    dataset = dataset.map(transform_example, remove_columns=["query", "labels"])

    # Save the tokenized dataset
    # Create the folder if it doesn't exist
    if not os.path.exists(config['tokenized_data_path']):
        os.makedirs(config['tokenized_data_path'])

    dataset.save_to_disk(config['tokenized_data_path'])

    return dataset


def setup_model_and_training(config: dict, bnb_config: BitsAndBytesConfig, ia3_config: IA3Config):
    # Initialize the accelerator and quantization configs
    model = AutoModelForCausalLM.from_pretrained(config['base_model_id'], quantization_config=bnb_config)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['base_model_id'])
    tokenizer.pad_token = tokenizer.eos_token

    # Set up model for training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, ia3_config)

    print({"trainable_params": model.print_trainable_parameters()})
    train_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['batch_size'],
        warmup_steps=5,
        gradient_checkpointing=False,
        max_steps=2000,
        learning_rate=5.0e-5,  # Want about 10x smaller than the Mistral learning rate
        logging_steps=config["eval_steps"],
        optim="paged_adamw_8bit",
        save_strategy="steps",  # Save the model checkpoint every logging step
        save_steps=config["eval_steps"],  # Save checkpoints every 50 steps
        evaluation_strategy="steps",  # Evaluate the model every logging step
        eval_steps=config["eval_steps"],  # Evaluate and save checkpoints every 50 steps
        do_eval=True,
        report_to=["wandb"],
        eval_accumulation_steps=1,
        run_name="proto0-1",
        load_best_model_at_end=True,
        bf16=torch.cuda.is_bf16_supported(),
        bf16_full_eval=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        fp16_full_eval=not torch.cuda.is_bf16_supported(),
    )

    return model, tokenizer, train_args


def create_all_path(config: dict):
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    if not os.path.exists(config['tokenized_data_path']):
        os.makedirs(config['tokenized_data_path'])

    if not os.path.exists(config['train_folder']):
        os.makedirs(config['train_folder'])

    if not os.path.exists(config['guidelines_folder']):
        os.makedirs(config['guidelines_folder'])

    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])
