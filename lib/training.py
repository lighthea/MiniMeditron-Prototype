import datasets
import requests
from datasets import Dataset

from lib.block import Transformer, OpenAITransformer, LocalTransformer
from lib.pipeline import Pipeline
import json
import os


# PipelineTrainer class is modified to use the Hugging Face Trainer logic
class PipelineTrainer:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def _dataset_to_openai_jsonl(self, hf_dataset: Dataset) -> str:
        """
        Transforms a Hugging Face dataset to the JSONL string format for OpenAI fine-tuning.

        :param hf_dataset: Hugging Face Dataset object.
        :return: JSONL formatted string.
        """
        jsonl_lines = []
        for example in hf_dataset:
            # Each example becomes a JSON object with "prompt" and "completion" keys
            json_object = {
                "prompt": example["text"].strip(),
                "completion": example["label"].strip()
            }
            # Convert the JSON object to a string and append it to the list
            jsonl_lines.append(json.dumps(json_object))

        # Join all JSON strings into a single newline-separated string (JSONL)
        jsonl_str = "\n".join(jsonl_lines)
        return jsonl_str

    def _create_hf_dataset_from_pairs(self, input_output_pairs: list) -> Dataset:
        """
        Creates a Hugging Face dataset from a list of input-output pairs.

        :param input_output_pairs: List of tuples, where each tuple is (input, output).
        :return: A Hugging Face Dataset object.
        """
        # Unzip the list of pairs into two separate lists
        inputs, outputs = zip(*input_output_pairs)

        # Create a dictionary suitable for creating a Dataset
        data_dict = {
            'text': list(inputs),
            'label': list(outputs)
        }

        # Create the Hugging Face Dataset
        dataset = Dataset.from_dict(data_dict)

        return dataset

    def _train_openai(self, block_to_train, training_data, model_params=None):
        block_name = block_to_train.name
        if block_to_train is None:
            raise ValueError(f"Block {block_name} does not exist in the pipeline.")

        transformed_training_data = self._dataset_to_openai_jsonl(training_data)
        # No need to preprocess src data for fine-tuning as OpenAI will handle it.
        openai_trainer = block_to_train.get_trainer(transformed_training_data, model_params)
        fine_tuning_result = openai_trainer.fine_tune(transformed_training_data)

        if 'model' in fine_tuning_result:  # Assuming 'model' key holds the fine-tuned model identifier
            block_to_train.update_model(fine_tuning_result['model'])

        return fine_tuning_result

    def _train_hf(self, block_to_train, train_dataset, compute_metrics=None, eval_dataset=None, training_args=None):
        block_name = block_to_train.name
        if block_to_train is None:
            raise ValueError(f"Block {block_name} does not exist in the pipeline.")
        
        # train_dataset = block_to_train.tokenizer(train_dataset)
        # eval_dataset = block_to_train.tokenizer(eval_dataset)

        # Now get the trainer from the Transformer block and start src
        hf_trainer = block_to_train.get_trainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            compute_metrics=compute_metrics
        )
        training_result = hf_trainer.train()

        # Optionally, we can update the model in the block with the newly trained model
        block_to_train.update_model(hf_trainer.model)

        return training_result

    def _handle_dataset(self, dataset):
        if isinstance(dataset, list):
            # Assume the dataset is a list of tuples (input, output)
            return self._create_hf_dataset_from_pairs(input_output_pairs=dataset)
        elif isinstance(dataset, datasets.Dataset):
            # Assume the dataset is a Hugging Face dataset
            return dataset

    def train(self, block_name, train_dataset, eval_dataset=None, training_args=None, compute_metrics=None):
        block_to_train = self.pipeline.blocks.get(block_name)
        if block_to_train is None:
            raise ValueError(f"Block {block_name} does not exist in the pipeline.")

        if not isinstance(block_to_train, Transformer):
            raise TypeError(f"Block {block_name} is not a Transformer and cannot be trained.")

        train_dataset = self._handle_dataset(train_dataset)
        truncated_pipeline = self.pipeline.get_dependency_strict_subpipeline(block_name)
        # Process the src data through the pipeline up to the block before the one to train
        processed_train_dataset = truncated_pipeline(train_dataset)
        processed_eval_dataset = None
        if eval_dataset is not None:
            eval_dataset = self._handle_dataset(eval_dataset)
            processed_eval_dataset = truncated_pipeline(eval_dataset)

        if isinstance(block_to_train, OpenAITransformer):
            self._train_openai(block_to_train, processed_train_dataset, model_params=training_args)
        elif isinstance(block_to_train, LocalTransformer):
            return self._train_hf(block_to_train,
                                  processed_train_dataset,
                                  processed_eval_dataset,
                                  training_args,
                                  compute_metrics)


def generate_task(guideline_folder, structure_file, model_name, output_dir, use_openai_api=False, examples=None):
    with open(structure_file, 'r') as file:
        name = json.load(file)['description']
    # Initialize the transformer
    if use_openai_api:
        transformer = OpenAITransformer(name=name, model_name=model_name, output_json=structure_file)
    else:
        transformer = LocalTransformer(name=name, model_name=model_name, output_json=structure_file, examples=examples)

    pipeline = Pipeline([transformer])
    # If examples are provided, use them to train the model
    if examples:
        if use_openai_api:
            pipeline_trainer = PipelineTrainer(pipeline)
            pipeline_trainer.train(transformer.name, examples)
        else:
            pass

    structured_guidelines = {}
    # Process each file in the guideline folder
    for filename in os.listdir(guideline_folder):
        if filename.endswith('.jsonl'):
            structured_guidelines[filename] = []
            file_path = os.path.join(guideline_folder, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    guideline = json.loads(line)
                    text = guideline['text']  # Assuming each object has a 'text' field
                    structured_output = transformer.forward(text)
                    structured_guidelines[filename].append(json.loads(structured_output))

            # Save the structured guidelines to a jsonl file
            with open(output_dir + filename, 'w') as f:
                for guideline in structured_guidelines[filename]:
                    f.write(json.dumps(guideline))
                    f.write('\n')

    return structured_guidelines
