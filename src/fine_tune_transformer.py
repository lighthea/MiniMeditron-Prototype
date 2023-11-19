import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.training import init_configs, setup_model_and_training, blanket, load_dataset, create_all_path, load_config, \
    init_wandb_project, launch_training


def main():
    # Load configuration
    conf_file = sys.argv[1]
    config = load_config(conf_file)

    # Create all paths
    create_all_path(config)

    # Initialize the wandb project
    init_wandb_project(config)

    # Initialize the accelerator and quantization configs
    bnb_config, ia3_conf = init_configs()

    # Set up model for training
    model, tokenizer, train_args = setup_model_and_training(config, bnb_config, ia3_conf)

    # Load the dataset
    dataset = load_dataset(config, tokenizer, blanket(config))

    # Initialize the trainer
    trainer = launch_training(model, tokenizer, train_args, dataset, ia3_conf)

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
