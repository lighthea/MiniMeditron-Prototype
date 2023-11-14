import sys

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from peft import get_peft_model
import torch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullOptimStateDictConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import wandb, os
from peft import IA3Config
from peft import prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
from tqdm import tqdm
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from scripts.tf_idf import *
from lib.metrics import *

wandb.login(key="a51dc03985c11e74e9ef700cd3093e6c78636177")

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)


accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

wandb_project = "minimed-finetune-proto0"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_LOG_MODEL"] = "end"  # log all model checkpoints

sys.path.append(os.path.join(current_dir, '..'))

data_folder = current_dir + "/../data/structured_patients"
guideline_folder = current_dir + "/../data/all_split_structured_guidelines"
tf_idf_path = current_dir + "/../data/TF-IDF"
folder_path = current_dir + "/../data/structured_patients/"
for files in tqdm(os.listdir(folder_path)):
    if files.endswith(".jsonl"):
        # chunk the file into multiple json files
        with open(folder_path + files) as f:
            for i, line in enumerate(f):
                with open(folder_path + files[:-6] + str(i) + '.json', 'w') as outfile:
                    json.dump(json.loads(line), outfile)
dataset = []

for file in tqdm(os.listdir(folder_path)):
    if file.endswith(".json"):
        # read the json file
        with open(folder_path + file) as f:
            data = json.load(f)
        # cut the data
        text = data["structured_patient"]
        label = data["condition_name"]

        dataset.append((text, label))

tf_idf_matrix, vectorizer = create_matrix(tf_idf_path, guideline_folder)

dataset = [(retrieve_top_k_guidelines(query, tf_idf_matrix, vectorizer, data_folder, k=3) + query, label) for
           (query, label) in dataset]
inputs, outputs = zip(*dataset)

# Create a dictionary suitable for creating a Dataset
data_dict = {
    'text': list(inputs),
    'label': list(outputs)
}

# Create the Hugging Face Dataset
dataset = Dataset.from_dict(data_dict)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
if not torch.cuda.is_bf16_supported():
    bnb_config.bnb_4bit_compute_dtype = torch.float16

base_model_id = "HuggingFaceH4/zephyr-7b-beta"
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


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
model = get_peft_model(model, ia3_config)
print(model.print_trainable_parameters())
model = accelerator.prepare_model(model)

# Check if gpu supports bf16
train_args = TrainingArguments(
    output_dir="./test",
    num_train_epochs=2,
    warmup_steps=5,
    per_device_train_batch_size=2,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
    max_steps=1000,
    learning_rate=5.0e-5,  # Want about 10x smaller than the Mistral learning rate
    logging_steps=50,
    optim="paged_adamw_8bit",
    save_strategy="steps",  # Save the model checkpoint every logging step
    save_steps=50,  # Save checkpoints every 50 steps
    evaluation_strategy="steps",  # Evaluate the model every logging step
    eval_steps=5,  # Evaluate and save checkpoints every 50 steps
    do_eval=True,
    report_to="wandb",
    run_name="proto0-1",
    load_best_model_at_end=True,
    logging_dir="./logs")

if torch.cuda.is_bf16_supported():
    train_args.bf16 = True
    train_args.bf16_full_eval = False

else:
    train_args.fp16 = True
    train_args.fp16_full_eval = True

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(100)),
    dataset_text_field="text",
    peft_config=ia3_config,
    compute_metrics=exact_matching,
    args=train_args
)

trainer.train()
