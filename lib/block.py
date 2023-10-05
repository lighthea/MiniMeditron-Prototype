import json

import dspy


class Structuriser(dspy.Module):
    def __init__(self, structure_file: str):
        """
        This module takes a structure file and outputs a structured data from free form text
        The structure file should contain :
            - A "document_structure" key with arbitrary sub objects representing the structure of the document
            - (Optional) A "prompt" key a prompt that is given to the model to fill in the structure (or defaults)
            - A "type" key that describes the purpose of the structure (e.g. "description of a patient", "diagnostic",
            "treatment")
        :param structure_file: The path to the structure file (JSON)
        """
        super().__init__()
        # Safely load the structure file
        try:
            structure_file = json.load(open(structure_file))
        except json.JSONDecodeError:
            raise ValueError("The structure file is not a valid JSON file.")
        # Check that the structure file has the required keys
        if "document_structure" not in structure_file:
            raise ValueError("The structure file does not contain a document structure.")
        if "type" not in structure_file:
            raise ValueError("The structure file does not contain a type.")
        # Set the type of the structure
        self.type: str = structure_file["type"]
        # Set the prompt of the structure
        if "prompt" in structure_file:
            self.prompt: str = structure_file["prompt"]
        else:
            self.prompt: str = "Complete this structure with the relevant information."

        # Set the structure
        self.structure: dict = structure_file["document_structure"]
        # Describe the signature of the output
        self.arg1: str = f"a {self.type}"
        self.arg2: str = "structure to follow"
        self.output: str = "completed structure from context"
        self.structuriser: dspy.Predict = dspy.Predict(
            f"{self.arg1}, {self.arg2} -> {self.output}")

    def forward(self, context: str):
        """
        This function takes a context and outputs a structured document
        :param context: The context of the structured document
        :return: A structured document
        """

        kwargs = {self.arg1: context, self.arg2: self.structure}
        return self.structuriser(**kwargs)


class MultiLabeledClassifierWithCommentary(dspy.Module):
    def __init__(self, output_structure_file: str):
        """
        This module takes a structure file and outputs a structured data with commentary using structured data as input
        The structure file should contain :
            - A "output_structure" key with arbitrary sub objects representing the fields to predict of the form :
                {output_1_name, output_2_name, ...}
            - (Optional) A "description" key to use for knowledge distillation
            - (Optional) A "task" key to use for knowledge distillation
        :param output_structure_file: the file containing the output structure
        """
        super().__init__()
        # Safely load the structure file
        try:
            output_structure_file = json.load(open(output_structure_file))
        except json.JSONDecodeError:
            raise ValueError("The structure file is not a valid JSON file.")
        # Check that the structure file has the required keys
        if "output_structure" not in output_structure_file:
            raise ValueError("The structure file does not contain an output structure.")
        # Set the output structure
        self.output_structure: dict = output_structure_file["output_structure"]
        # Check that the output structure is valid
        if not all(isinstance(output, str) for output in self.output_structure.values()):
            raise ValueError("The output structure is not a dict of output names.")

        # Set the description
        if "description" in output_structure_file:
            self.description: str = output_structure_file["description"]
        else:
            self.description: str = "Predict the following fields."

        # Retrieve the actual name of the task if it exists
        if "task_name" in output_structure_file:
            self.task_name: str = output_structure_file["task"]
        else:
            self.task_name: str = "task"
        # Infer the outputs names of the model
        self.output_names: list = list(self.output_structure.keys())
        # Describe the signature of the output
        self.arg1: str = f"structured data for {self.task_name}"
        self.outputs: str = f"{', '.join(self.output_names)}"
        self.classifier: dspy.Predict = dspy.Predict(
            f"{self.arg1} -> {self.outputs}")

    def forward(self, structured_data: dict):
        """
        This function takes a structured data and outputs a structured data with commentary
        :param structured_data: The structured data
        :return: A structured data with commentary
        """
        kwarg = {self.arg1: str(structured_data)}
        return self.classifier(**kwarg)


class Verbaliser(dspy.Module):
    def __init__(self, format_file: str):
        """
        This module takes a file describing a structure and instruction on how to verbalise it to write a free form text
        The file should contain :
            - A "expanded_desc" key that describes how to expand the given structure
        :param format_file: The path to the format file (JSON)

        """
        super().__init__()
        # Safely load the format file
        try:
            format_file = json.load(open(format_file))
        except json.JSONDecodeError:
            raise ValueError("The format file is not a valid JSON file.")
        # Check that the format file has the required keys
        if "expanded_desc" not in format_file:
            raise ValueError("The format file does not contain a description for expansion.")

        # Set the expanded description
        self.expanded_desc: str = format_file["expanded_desc"]
        # Describe the signature of the output
        self.arg1: str = self.expanded_desc
        self.output: str = "a free form text of the structured document"
        self.verbaliser: dspy.Predict = dspy.Predict(
            f"{self.arg1} -> {self.output}")

    def forward(self, structured_data: dict):
        """
        This function takes a structured data and outputs a free form text
        :param structured_data: The structured data
        :return: A free form text
        """
        kwarg = {self.arg1: str(structured_data)}
        return self.verbaliser(**kwarg)
