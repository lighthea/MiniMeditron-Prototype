import json
from transformers import Trainer

class Pipeline():
    def __init__(self, blocks: list):
        '''
        This is a pipeline. It contains a list of blocks. Each block has a name, a previous block and a next block.
        The chain of block prepresent the whole process of the pipeline.
        '''
        self.blocks = blocks
        # each element of blocks is a dict containing : 
        # name : name of the block
        # previous : the previous block
        # next : the next block
        # Block : the block itself

        # maybe we can read from a config file to construct the pipeline (read the name, type of block etc...?)

    def forward(self):
        '''
        Forward the input through the pipeline.
        '''
        for block in self.blocks:
            if block['next'] == None:
                return block['Block'].forward()
            else:
                new_input = block['Block'].forward()
                block['next'].change_input(new_input)

    def remove_block(self, block_name):
        '''
        Remove the block with name block_name from the pipeline.
        '''
        for block, i in self.blocks:
            if block['name'] == block_name:
                self.blocks[i-1]['next'] = block['next']
                self.blocks[i+1]['previous'] = block['previous']
                self.blocks = self.blocks[:i] + self.blocks[i+1:]

class PipelineTrainer(Trainer):
    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline

    def train(self, block_name):
        # train the block with name block_name
        self.pipeline.blocks[block_name] # call the TRAIN function ? how ?



