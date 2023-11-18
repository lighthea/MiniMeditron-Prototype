import os
import pandas as pd
from transformers.integrations import WandbCallback
import wandb

from lib.utils import decode_predictions


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each logging step during training.
    It allows to visualize the model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset for generating predictions.
        num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=1):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions every `freq` epochs
        if state.global_step % state.eval_steps == 0 and state.eval_steps % self.freq == 0:
            # generate predictions
            predictions = self.trainer.predict(self.sample_dataset)
            # decode predictions and labels
            predictions = decode_predictions(self.tokenizer, predictions)
            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # log the table to wandb
            self._wandb.log({"sample_predictions": records_table})


def retrieve_last_wandb_run_id(config: dict) -> str | None:
    # Initialize the API client
    api = wandb.Api()

    # Replace 'your_username' with your wandb username and 'your_project_name' with your project name

    runs = api.runs(f"alexs-team/{config['wandb_project']}")
    if len(runs) == 0:
        return None
    # The first run in the list is the most recent one
    return runs[0].id


def init_wandb_project(config: dict) -> None:
    # Wandb Login
    print("Logging into wandb")
    wandb.login(key=config['wandb_key'])

    if len(config["wandb_project"]) > 0:
        os.environ["WANDB_PROJECT"] = config["wandb_project"]
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def retrieve_checkpoint(config: dict) -> str | None:
    last_run_id = retrieve_last_wandb_run_id(config)
    if last_run_id is None:
        print("No checkpoint found")
        return None
    # If there already exists a checkpoint for this model in wandb then retrieve it
    with wandb.init(
            project=os.environ["WANDB_PROJECT"],
            id=last_run_id,
            resume="must", ) as run:
        # Connect an Artifact to the run
        my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
        my_checkpoint_artifact = run.use_artifact(my_checkpoint_name)

        # Download checkpoint to a folder and return the path
        checkpoint_dir = my_checkpoint_artifact.download()

        return checkpoint_dir
