from lib.block import *
from lib.pipeline import *
from lib.training import *

openAI_model = OpenAITransformer(name="Model TMP : Patient Generator",
                                 model_name="gpt3.5-turbo",
                                 output_json="data/structure/pipelines/tasks/patient_generation.json")
patientGenerationPipeline = Pipeline([openAI_model])
trainer = PipelineTrainer(patientGenerationPipeline)

trainer.train("Model TMP : Patient Generator", "")