import sys
import os
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.training import init_configs, setup_model_and_training_finetuning, \
    load_config, \
    init_wandb_project, launch_training, launch_training_qa, launch_training_finetune
from lib.wandb import retrieve_checkpoint  
  

def main():
    # Retrieve all config file paths from command line arguments
    config_files = sys.argv[1:]  # This will take all arguments except the script name
    bnb_config, ia3_conf = init_configs()
    config = load_config(config_files[0])
    model, tokenizer, _ = setup_model_and_training_finetuning(config, bnb_config, ia3_conf)
    adapter_names = []
    # Loop over each configuration file and run the pipeline
    for i, conf_file in enumerate(config_files[:-1]):
      config = load_config(conf_file)
      name = f"adapter {i}"
      adapter_names.append(name)
      print(conf_file)
      adapter = retrieve_checkpoint(config)
      model.load_adapter(adapter, adapter_name = name)
    with open(config_files[-1], "r") as f:
      input_text = f.read()
    for adapter in adapter_names:
      print(input_text)
      model.set_adapter(adapter)
      tokenized_input = tokenizer(input_text, return_tensors='pt')
      output = model.generate(**tokenized_input)
      input_text = tokenizer.decode(output[0], skip_special_tokens=True)
      

  
    

    
if __name__ == "__main__":
    main()
