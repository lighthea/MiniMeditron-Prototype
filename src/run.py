import sys

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, EvalPrediction, \
    DataCollatorWithPadding, DataCollatorForLanguageModeling
import wandb, os, torch, json
from tqdm import tqdm
from peft import IA3Config, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
from evaluate import load
from functools import partial

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.tf_idf import batch_bm25
from lib.utils import retrieve_prompt, init_wandb_project

# Init the metric
exact_matching = load("exact_match")


def blanket(config: dict) -> str:
    file = config['process_file']
    with open(file, 'r') as f:
        data = json.load(f)
    return data["prompt"].replace('""', "LABEL")


def compute_metrics(eval_pred: EvalPrediction, tokenizer, blanket):
    predictions, label_ids = eval_pred
    decoded_prediction = [tokenizer.decode(prediction, skip_special_tokens=True) for prediction in predictions]
    decoded_labels = [blanket.replace("LABEL", tokenizer.batch_decode(label_id, skip_special_tokens=True)) for label_id in label_ids]

    print(decoded_prediction[0])
    # Calculate exact match
    results = exact_matching.compute(predictions=decoded_prediction, references=decoded_labels)
    return results


def init_configs(bf16_support: bool):
    float_type = torch.bfloat16 if bf16_support else torch.float16
    print(f"Using {float_type} for training")
    print("Initializing accelerator and quantization configs")

    # Initialize the quantization config

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=float_type
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


# Pre-tokenization Function
def load_dataset(config: dict, tokenizer, blanket: str) -> [dict]:
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
    dataset = batch_bm25(dataset, config['guidelines_folder'], n=config['n_context_guidelines'], base_folder=config['base_folder'])

    # Merge the query and context into a single string using the prompt defined in the structure file
    partial_prompt = retrieve_prompt(config)
    dataset = dataset.map(lambda x: {"query": partial_prompt
                          .replace("INPUT", str(x["text"]))
                          .replace("CONTEXT", str(x["context"]))}
                          , remove_columns=["text", "context"])

    # Tokenize the dataset
    def transform_example(example):
        tokenized_output = {"text": tokenizer.apply_chat_template([
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": blanket.replace("LABEL", example["labels"])}
        ], tokenize=False, padding="max_length", add_generation_prompt=False)}

        # Convert tensor output to a dictionary format suitable for the dataset
        return tokenized_output

    dataset = dataset.map(transform_example,remove_columns=["query", "labels"])

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
        logging_steps=60,
        optim="paged_adamw_8bit",
        save_strategy="steps",  # Save the model checkpoint every logging step
        save_steps=60,  # Save checkpoints every 50 steps
        evaluation_strategy="steps",  # Evaluate the model every logging step
        eval_steps=config["eval_steps"],  # Evaluate and save checkpoints every 50 steps
        do_eval=True,
        report_to=["wandb"],
        eval_accumulation_steps=2,
        run_name="proto0-1",
        load_best_model_at_end=True,
        bf16=torch.cuda.is_bf16_supported(),
        bf16_full_eval=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        fp16_full_eval=not torch.cuda.is_bf16_supported(),
    )

    return model, tokenizer, train_args


def main():
    # Load configuration
    with open('conf/config_train_m2.json') as config_file:
        config = json.load(config_file)

    # Initialize the wandb project
    init_wandb_project(config)

    # Initialize the accelerator and quantization configs
    bnb_config, ia3_conf = init_configs(torch.cuda.is_bf16_supported())

    # Set up model for training
    model, tokenizer, train_args = setup_model_and_training(config, bnb_config, ia3_conf)

    # Load the dataset
    dataset = load_dataset(config, tokenizer, blanket(config))
    print(dataset["text"][0])
    # Randomize the dataset and split into train and validation sets
    dataset = dataset.shuffle()
    dataset = dataset.train_test_split(test_size=0.01, shuffle=True)

    compute_metrics_with_tokenizer = partial(compute_metrics, tokenizer=tokenizer, blanket=blanket(config))
    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=ia3_conf,
        dataset_text_field="text",
        #compute_metrics=compute_metrics_with_tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
