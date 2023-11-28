import json
import os

import torch
import wandb
from accelerate import Accelerator
from peft import IA3Config, LoraConfig, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.modeling_utils import unwrap_model
from transformers.utils import is_peft_available
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from .secure_env import read_secure_file
from .dataset import load_extras

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, DPOTrainer
from lib.wandb import retrieve_checkpoint

class TfIdfTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(TfIdfTrainer, self).__init__(*args, **kwargs)

    def initialize_metric(self, dataset, config, tokenizer):
        # Load guidelines
        extras = load_extras(config)

        # Build tf-idf matrix
        

        print(extras)
        raise NotImplementedError()

    def compute_loss(self, model, inputs, return_outputs=False):
        # This code is totally mine and not a copy paste from https://github.com/huggingface/transformers/blob/ce315081340fdf6846f16c321eb53878b6272d53/src/transformers/trainer.py
        # Get label and predication tokens
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

METRIC_LOSS_DICT = {
    "accuracy": SFTTrainer,
    "tfidf": TfIdfTrainer
}

def init_configs(config):
    bf16_support = torch.cuda.is_bf16_supported()
    float_type = torch.bfloat16 if bf16_support else torch.float16
    print(f"Using {float_type} for training")
    print("Initializing accelerator and quantization configs")

    # Initialize the quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # Initialize the IA3 config
    if config["wandb_parameters"]["start_from_checkpoint"]:
        config["chekpoint_folder"] = retrieve_checkpoint(config)
        # ia3_config = IA3Config.from_pretrained(config["chekpoint_folder"])

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
        feedforward_modules=["down_proj"],
        init_ia3_weights=not config["wandb_parameters"]["start_from_checkpoint"],
    )

    return bnb_config, ia3_config


def setup_model_and_training_finetuning(config: dict, bnb_config: BitsAndBytesConfig, ia3_config: IA3Config):
    # Initialize the accelerator and quantization configs
    if config["wandb_parameters"]["start_from_checkpoint"]:
        model = AutoModelForCausalLM.from_pretrained(config["chekpoint_folder"],
                                                     quantization_config=bnb_config,
                                                     use_flash_attention_2=config["general_settings"]["use_flash_attn"]
                                                     , device_map={"": Accelerator().process_index})
        tokenizer = AutoTokenizer.from_pretrained(config["chekpoint_folder"], add_eos_token=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(config["general_settings"]['base_model_id'],
                                                     quantization_config=bnb_config,
                                                     use_flash_attention_2=config["general_settings"]["use_flash_attn"]
                                                     , device_map={"": Accelerator().process_index})

        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["general_settings"]['base_model_id'], add_eos_token=True)
        tokenizer.pad_token = tokenizer.eos_token

    # Set up model for training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config["model_parameters"][
        "gradient_checkpointing"])
    # model = get_peft_model(model, ia3_config)
    train_args = TrainingArguments(
        output_dir=config["model_folders"]['output_dir'],
        warmup_steps=5,
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
    content = read_secure_file(config_file)
    config = json.loads(content)

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
    match config["general_settings"]["task"]:
        case "qa":
            return launch_training_qa(model, tokenizer, train_args, dataset, ia3_conf, config)
        
        case "finetune":
            return launch_training_finetune(model, tokenizer, train_args, dataset, ia3_conf, config)
    
        case "po":
            return launch_training_po(model, tokenizer, train_args, dataset, ia3_conf, config)

        case _:
            return Exception("Unrecognized value for task: {}".format(config["general_settings"]["task"]))

def launch_training_finetune(model, tokenizer, train_args, dataset, ia3_conf, config):
    tokenizer.padding_side = "right"
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

def launch_training_po(model, tokenizer, train_args, dataset, ia3_conf, config):
    tokenizer.padding_side = "left"
    # Print dataset structure
    print(dataset["train"])
    # Determine max seq length
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
        truncation_mode="keep_end",
        generate_during_eval=False
    )

    return trainer

def launch_training_qa(model, tokenizer, train_args, dataset, ia3_conf, config):
    # List of metrics
    loss_type = config["loss"]["type"]
    if not loss_type in METRIC_LOSS_DICT:
        raise Exception("unrecognized loss type {}".format(loss_type))
    trainer_builder = METRIC_LOSS_DICT[loss_type]
    
    instruction_template = "<|user|>"
    response_template = "<|assistant|>"
    tokenizer.padding_side = "right"

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

    trainer = trainer_builder(
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

    if hasattr(trainer, 'initialize_metric'):
        trainer.initialize_metric(dataset, config, tokenizer)

    return trainer
