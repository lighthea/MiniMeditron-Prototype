import json


class Block():
    def __init__(self, resources, output_json, model, name):
        '''
        This is a block of the pipeline. It contains a resource path (that act as a knowledge database is the block is a retriever), a model, an output_json file and a name.
        '''
        self.resources = resources
        try: 
            self.output_json = json.load(open(output_json))
        except json.JSONDecodeError:
            raise ValueError("The output_json file is not a valid JSON file.")
        self.model = model
        self.name = name

    def forward(self, inputs):
        # fill the output_json with the inputs at the correct place
        prompt = self.fill(self.output_json, inputs)
        return self.model(prompt)
    
    def fill(self, output_json, inputs):
        # fill the output_json with the inputs at the correct place
        # scoot where to put the inputs in the outpu_json file
        return output_json
    
    def change_input(self, new_input):
        self.output_json = new_input