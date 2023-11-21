import json
import os

import torch
import wandb
from datasets import Dataset, DatasetDict
from peft import IA3Config, prepare_model_for_kbit_training, get_peft_model, prepare_model_for_int8_training
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from lib.tf_idf import batch_bm25
from lib.utils import retrieve_prompt


def blanket(config: dict) -> str:
    file = config["model_parameters"]['process_file']
    with open(file, 'r') as f:
        data = json.load(f)

    data["document_structure"]["Condition"] = "LABEL"
    return json.dumps(data["document_structure"])


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


def load_dataset(config: dict,
                 tokenizer,
                 blanket_string: str = None,
                 with_context: bool = True,
                 with_output: bool = True,
                 with_token: bool = False) -> DatasetDict:
    # Check if the dataset has already been tokenized
    print("Checking if tokenized dataset exists")
    if (os.path.exists(config["model_folders"]['tokenized_data_path']) and
            not config["other_parameters"]['force_retokenize']):
        # check if directory is not empty
        if os.listdir(config["model_folders"]['tokenized_data_path']):
            print("Loading tokenized dataset")
            # Load the tokenized dataset
            dataset = DatasetDict.load_from_disk(config["model_folders"]['tokenized_data_path'])
            return dataset

    # For each patient, retrieve the top k guidelines
    queries = []
    labels = []
    for file in tqdm(os.listdir(config["general_folders"]['train_folder']), desc="Loading dataset"):
        if file.endswith(".jsonl"):
            with open(os.path.join(config["general_folders"]['train_folder'], file)) as f:
                for line in f:
                    data = json.loads(line)
                    queries.append(data["structure"])
                    labels.append(data["condition_name"])  # TODO: change to labels using utils.replace_string_in_files

    dataset = Dataset.from_dict({"text": queries, "labels": labels})
    del queries, labels

    # Append the guidelines to the dataset if needed
    if with_context:
        dataset = batch_bm25(dataset, config["general_folders"]['guidelines_folder'],
                             n=config["model_parameters"]['n_context_guidelines'],
                             base_folder=config["general_folders"]['base_folder'])

    # Merge the query and context into a single string using the prompt defined in the structure file
    partial_prompt = retrieve_prompt(config["model_parameters"]['process_file'])

    if with_context:
        dataset = dataset.map(lambda x: {"query": partial_prompt
                              .replace("INPUT", str(x["text"]))
                              .replace("CONTEXT", str(x["context"]))}
                              , remove_columns=["text", "context"])
    else:
        dataset = dataset.map(lambda x: {"query": partial_prompt
                              .replace("INPUT", str(x["text"]))}
                              , remove_columns=["text"])

    # Tokenize the dataset
    def transform_example(example):
        assistant_prompt = ""
        if with_output:
            if blanket_string is None:
                assistant_prompt = example["labels"]
            else:
                assistant_prompt = blanket_string.replace("LABEL", example["labels"])
        if with_output:
            tokenized_output = {"text": tokenizer.apply_chat_template([
                {"role": "user", "content": example["query"]},
                {"role": "assistant", "content": assistant_prompt}
            ], tokenize=False,  add_generation_prompt=False)}

        else:
            tokenized_output = {"text": tokenizer.apply_chat_template([
                {"role": "user", "content": example["query"]},
            ], tokenize=False, add_generation_prompt=False)}

        return tokenized_output

    def tokenize(example):
        tokenized = tokenizer.encode(example["text"], add_special_tokens=True,
                                     padding="max_length")
        return {"input_ids": tokenized}

    dataset = dataset.map(transform_example, remove_columns=["query", "labels"])
    if with_token:
        dataset = dataset.map(tokenize)

    dataset = dataset.shuffle()
    dataset = dataset.train_test_split(test_size=config["model_parameters"]["test_size"], shuffle=True)

    # Create the folder if it doesn't exist
    if not os.path.exists(config["model_folders"]['tokenized_data_path']):
        os.makedirs(config["model_folders"]['tokenized_data_path'])

    # Save the tokenized dataset
    dataset.save_to_disk(config["model_folders"]['tokenized_data_path'])

    return dataset


def setup_model_and_training_finetuning(config: dict, bnb_config: BitsAndBytesConfig, ia3_config: IA3Config):
    # Initialize the accelerator and quantization configs
    model = AutoModelForCausalLM.from_pretrained(config["model_parameters"]['base_model_id'],
                                                 quantization_config=bnb_config,
                                                 use_flash_attention_2=True
                                                 )

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_parameters"]['base_model_id'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Set up model for training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, ia3_config)

    print({"trainable_params": model.print_trainable_parameters()})
    train_args = TrainingArguments(
        output_dir=config["model_folders"]['output_dir'],
        num_train_epochs=config["model_parameters"]['num_train_epochs'],
        per_device_train_batch_size=config["model_parameters"]['batch_size'],
        warmup_steps=50,
        gradient_checkpointing=False,
        max_steps=config["model_parameters"]['max_steps'],
        learning_rate=config["model_parameters"]["learning_rate"],
        # Want about 10x smaller than the Mistral learning rate
        logging_steps=config["model_parameters"]["logging_steps"],
        optim="paged_adamw_8bit",
        save_strategy="steps",  # Save the model checkpoint every logging step
        save_steps=config["model_parameters"]["eval_steps"],  # Save checkpoints every 50 steps
        evaluation_strategy="steps",  # Evaluate the model every logging step
        eval_steps=config["model_parameters"]["eval_steps"],  # Evaluate and save checkpoints every 50 steps
        do_eval=True,
            report_to=["wandb"],
        eval_accumulation_steps=1,
        run_name="proto0-1",
        neftune_noise_alpha=5,
        load_best_model_at_end=True,
        bf16=torch.cuda.is_bf16_supported(),
        bf16_full_eval=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        fp16_full_eval=not torch.cuda.is_bf16_supported(),
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
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def launch_training(model, tokenizer, train_args, dataset, ia3_conf):
    max_seq_length = 0
    for example in tqdm(dataset["train"], desc="Estimating max seq length"):
        max_seq_length = max(max_seq_length, len(tokenizer.encode(example["text"])))
    print(f"Max seq length: {max_seq_length}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=ia3_conf,
        dataset_text_field="text",
        neftune_noise_alpha=5,
        #max_seq_length=max_seq_length+1,
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
    max_seq_length = 0
    for example in tqdm(dataset["train"], desc="Estimating max seq length"):
        max_seq_length = max(max_seq_length, len(tokenizer.encode(example["text"])))
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
        max_seq_length=max_seq_length+1,
    )

    return trainer
