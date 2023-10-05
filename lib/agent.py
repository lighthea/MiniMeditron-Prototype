import dspy
import json


class Agent(dspy.Module):
    def __init__(self, config_file: str):
        """
        An agent takes a free form text as input and outputs free form text, it is composed by a graph of submodules.
        This module takes a config file describing the agent and its modules.
        The config file should contain :
            - A "modules" key that describes the modules of the agent
                Each module should have a "name" key that describes the name of the module, and the name of the modules
                linked to each of its inputs
            - A "description" key that informs humans on the goal of this agent
            - A "output" :key that gives the different endpoints of this agent
        :param config_file: the path to the config file (JSON)
        """
        super().__init__()
        # Safely load the config file
        try:
            config_file = json.load(open(config_file))
        except json.JSONDecodeError:
            raise ValueError("The config file is not a valid JSON file.")
