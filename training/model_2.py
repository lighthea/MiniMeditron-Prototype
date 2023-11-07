import numpy as np
from lib.pipeline import *
from lib.block import *
from lib.training import *

Model1 = LocalTransformer(name="Model 1 : Patient Structuriser",
                          output_json="/home/sallinen/Programmation/minimeditron/data/structure/pipelines/pipeline_1"
                                      ".json")
Model2 = LocalTransformer(name="Model 2 : Diagnoser",
                          output_json="/home/sallinen/Programmation/minimeditron/data/structure/pipelines/pipeline_2"
                                      ".json")
Model3 = LocalTransformer(name="Model 3 : Treatment",
                          output_json="/home/sallinen/Programmation/minimeditron/data/structure/pipelines/pipeline_3"
                                      ".json")
Model4 = LocalTransformer(name="Model 4 : Report maker",
                          output_json="/home/sallinen/Programmation/minimeditron/data/structure/pipelines/pipeline_4"
                                      ".json")

# GuidelineRetriever = SimpleRetriever(name="Guideline Retriever",
# resources="/home/sallinen/Programmation/minimeditron/data/processed/")

# pipeline = Pipeline(Model1 > [GuidelineRetriever > Model2, Model2 > Model3 > Model4])
_ = Model1 > Model2 > Model3 > Model4

pipeline = Pipeline([Model1, Model2, Model3, Model4])
print(pipeline("Patient has a headache and a fever."))

print(pipeline.get_dependency_subpipeline("Model 1 : Patient Structuriser").block_sequence)
em = Pipeline([])
print(em.add_block(Model2).block_sequence)