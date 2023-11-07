from datasets import Dataset

from lib.block import Block


class Pipeline:
    def __init__(self, blocks):
        """
        This is a pipeline. It contains a list of Block instances which represent the stages in the pipeline.
        The blocks are connected based on their names using a dictionary.
        """
        if not blocks or len(blocks) == 0:
            self.blocks = {}
            self.start_blocks = []
            self.end_blocks = []
            self.block_sequence = []
        else:
            if not all(isinstance(block, Block) for block in blocks):
                raise ValueError("All pipeline elements must be instances of Block.")

            # Flatten the list in case of nested lists due to chaining
            flat_blocks = self._flatten_blocks(blocks)
            self.blocks = {block.name: block for block in flat_blocks}
            self.start_blocks = [block.name for block in flat_blocks if
                                 (not block.previous_block_names or len(block.previous_block_names) == 0)]
            self.end_blocks = [block.name for block in flat_blocks if
                               (not block.next_block_names or len(block.next_block_names) == 0)]
            self.block_sequence = self._construct_sequence(flat_blocks)

    def _flatten_blocks(self, blocks):
        flat_list = []
        for block in blocks:
            if isinstance(block, list):
                flat_list.extend(self._flatten_blocks(block))
            else:
                flat_list.append(block)
        return flat_list

    def _construct_sequence(self, blocks):
        if not blocks or len(blocks) == 0:
            return []
        # Construct the execution sequence based on topological sorting
        # This assumes there are no circular dependencies between blocks
        # Create a mapping from block name to Block instance
        block_map = {block.name: block for block in blocks}

        # Initialize all vertices with no incoming edge
        in_degree = {block: 0 for block in block_map.values()}
        # Populate in_degree counts
        for block in blocks:
            if block.previous_block_names:
                in_degree[block] += len(block.previous_block_names) - 1

        # Create a set of all blocks with no incoming edges (start nodes for the sort)
        start_blocks = [block for block in block_map.values() if in_degree[block] == 0]
        sorted_blocks = []
        while start_blocks:
            # Select a block from start_blocks and remove it
            n = start_blocks.pop()
            sorted_blocks.append(n)
            # Decrease in_degree by 1 for all neighbors (successor blocks)
            for neighbor_name in n.next_block_names:
                neighbor = block_map[neighbor_name]
                in_degree[neighbor] -= 1
                # If in_degree becomes 0, add it to start_blocks
                if in_degree[neighbor] == 0:
                    start_blocks.append(neighbor)

        # Check for an error (which would indicate a cycle, and thus, an invalid DAG)
        if len(sorted_blocks) != len(blocks):
            raise ValueError("There is a cycle in the block dependencies.")

        return [block.name for block in sorted_blocks]

    def forward(self, input_data):
        """
        Forward the input through the pipeline based on the constructed sequence.
        """

        if not self.blocks or len(self.blocks) == 0:
            return input_data

        # Dictionary to hold the output of each block
        block_outputs = {}
        # First, run all start blocks with the initial input
        for block_name in self.start_blocks:
            block = self.blocks[block_name]
            block_outputs[block_name] = block.forward(input_data)

        # Then, process the rest of the blocks in the sequence
        for block_name in reversed(self.block_sequence):
            if block_name in self.start_blocks:
                # Already processed as a start block
                continue

            block = self.blocks[block_name]
            # Gather inputs from previous blocks
            inputs_for_block = [block_outputs[prev_name] for prev_name in block.previous_block_names]
            # If there is only one input, pass it directly; otherwise, pass all as a tuple
            block_input = inputs_for_block[0] if len(inputs_for_block) == 1 else tuple(inputs_for_block)
            block_outputs[block_name] = block.forward(block_input)

        # Identify end blocks and collect their outputs
        results = {block_name: block_outputs[block_name] for block_name in self.end_blocks}

        return results

    def __call__(self, input_data):
        if isinstance(input_data, str):
            return self.forward(input_data)
        elif isinstance(input_data, Dataset):
            inputs = input_data['text']
            return [self.forward(input_) for input_ in inputs]
        elif isinstance(input_data, list):
            if all(isinstance(input_, (str, str)) for input_ in input_data):
                return [self.forward(input_) for (input_, output_) in input_data]
            elif all(isinstance(input_, str) for input_ in input_data):
                return [self.forward(input_) for input_ in input_data]
            else:
                raise TypeError("Input data is not a list of strings.")

    def get_dependency_subpipeline(self, block_name):
        if block_name not in self.block_sequence:
            raise ValueError(f"Block {block_name} is not in the pipeline.")
        new_selection = self.block_sequence[self.block_sequence.index(block_name):]
        blocks = [self.blocks[block_name] for block_name in new_selection]
        blocks[0].next_block_names = []
        return Pipeline(blocks)

    def get_dependency_strict_subpipeline(self, block_name):
        if block_name not in self.block_sequence:
            raise ValueError(f"Block {block_name} is not in the pipeline.")

        new_selection = self.block_sequence[self.block_sequence.index(block_name) + 1:]
        if new_selection:
            blocks = [self.blocks[block_name] for block_name in new_selection]
            blocks[0].next_block_names = []
            return Pipeline(blocks)
        else:
            return Pipeline([])

    def add_block(self, block, after_block_names: list[str] | int = -1):
        """
        Add a block to the pipeline. It will be placed after the block with after_block_name.
        """

        if not self.blocks or len(self.blocks) == 0:
            if block.previous_block_names:
                raise ValueError("Cannot add a block alone that has not its inputs met.")
            block.next_block_names = []
            self.blocks = {block.name: block}
            self.block_sequence = [block.name]
            self.start_blocks = [block.name]
            self.end_blocks = [block.name]
            return self

        if block.name in self.blocks:
            raise ValueError(f"Block {block.name} is already in the pipeline.")
        if (after_block_names and isinstance(after_block_names, list) and
                any(after_block_name not in self.blocks for after_block_name in after_block_names)):
            raise ValueError(f"Blocks are not in the pipeline.")
        if after_block_names:
            if isinstance(after_block_names, int):
                after_block_names = self.end_blocks
            # Set the new block's previous_block_names to after_block_name
            block.previous_block_names = after_block_names
            block.next_block_names = []
            for after_block_name in after_block_names:
                # Update the previous_block_names of the after_block to point to the new block
                self.blocks[after_block_name].next_block_names.append(block.name)

        # Add the new block to the blocks dict and sequence list
        self.blocks[block.name] = block
        # Topologically sort the sequence list
        self.block_sequence = self._construct_sequence(self.blocks.values())
        self.start_blocks = [block.name for block in self.blocks.values() if
                             (not block.previous_block_names or len(block.previous_block_names) == 0)]
        self.end_blocks = [block.name for block in self.blocks.values() if
                           (not block.next_block_names or len(block.next_block_names) == 0)]
        return self
