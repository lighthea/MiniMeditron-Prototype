import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from lib.block import *
from lib.pipeline import *
from lib.training import *
from lib.env import *
from lib.metrics import *
from lib.pipeline import Pipeline
from lib.block import LocalTransformer, Selector

transformer1 = LocalTransformer(name="Patient Structuriser",
                                model_name="mistralai/Mistral-7B-v0.1",
                                output_json="data/structure/pipelines/pipeline_1.json")

selector = Selector(resources="../../Guidelines/processed",
                    name="Resource Selector")

transformer2 = LocalTransformer(name="Diagnoser",
                                model_name="allenai/longformer-base-4096",
                                output_json="../data/structure/pipelines/pipeline_2.json")

_ = transformer1 > selector > transformer2
pipeline = Pipeline(transformer1, selector, transformer2)

# Example use
result = pipeline("Patient has a headache and a fever.")