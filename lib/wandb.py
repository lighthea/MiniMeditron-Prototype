import os
from transformers.integrations import WandbCallback
import wandb
from transformers import Trainer

from lib.utils import decode_predictions


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each logging step during training.
    It allows to visualize the model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        val_dataset (Dataset): A subset of the validation dataset for generating predictions.
        num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100):
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

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions every `freq` epochs
        # if state.global_step % state.eval_steps == 0:
        # generate predictions
        # print(list(self.trainer.get_eval_dataloader(self.sample_dataset)))
        # predictions = self.trainer.evaluate(self.sample_dataset)
        # decode predictions and labels
        # predictions = decode_predictions(self.tokenizer, predictions)
        # add predictions to a wandb.Table
        # predictions_df = pd.DataFrame(predictions)
        # predictions_df["step"] = state.global_step
        # records_table = self._wandb.Table(dataframe=predictions_df)
        # log the table to wandb
        # ._wandb.log({"sample_predictions": records_table})


def retrieve_last_wandb_run_id(config: dict) -> str | None:
    # Initialize the API client
    api = wandb.Api()

    runs = api.runs(f"alexs-team/{config['wandb_parameters']['wandb_project']}")
    if len(runs) == 0:
        return None
    # The first run in the list is the most recent one
    print(f"Found {len(runs)} runs")
    print(f"Last run id: {runs[0].id}")
    return runs[0].id


def retrieve_checkpoint(config: dict) -> str | None:
    if os.path.exists(os.path.join(config["wandb_parameters"]["wandb_folder"])):
        return config["wandb_parameters"]["wandb_folder"]

    run = wandb.init(entity="alexs-team", project=config["wandb_parameters"]["wandb_project"], name=config["wandb_parameters"]["run_name"], resume="allow")
    artifact = run.use_artifact(f'alexs-team/minimed-finetune-proto0/model-{str(config["wandb_parameters"]["baseline_name"])}:latest',
                                type='model')
    artifact_dir = artifact.download(config["wandb_parameters"]["wandb_folder"])
    return artifact_dir