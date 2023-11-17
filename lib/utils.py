import json
import os

import wandb
from tqdm import tqdm

def yield_structured_obj(folder):
    try:
        data = []
        for file in sorted(os.listdir(folder)):
            if file.endswith(".jsonl"):
                with open(file, 'r') as f:
                    for line in f:
                        data = json.loads(str(line))
                        data.append(str(data["structured_patient"]))
        return data

    except json.JSONDecodeError:
        # Handle the case where the input is not a valid JSON string
        raise ValueError("Invalid JSON string provided.")


def replace_string_in_files(folder_path, old_string, new_string):
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".jsonl"):  # Ensuring only structured files are processed
            file_path = os.path.join(folder_path, filename)

            # Read the contents of the file
            with open(file_path, 'r') as file:
                filedata = file.read()

            # Replace the target string
            filedata = filedata.replace(old_string, new_string)

            # Write the file out again
            with open(file_path, 'w') as file:
                file.write(filedata)


def retrieve_prompt(config: dict) -> str:
    file = config['process_file']
    with open(file, 'r') as f:
        data = json.load(f)
    return data["prompt"].replace("OUTPUT", str(data["document_structure"]))


def retrieve_last_wandb_run_id(config: dict) -> str:
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
