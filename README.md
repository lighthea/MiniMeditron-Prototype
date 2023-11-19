# Minimeditron-Prototype

# Run on cluster 

cd ~ && wget "https://raw.githubusercontent.com/lighthea/MiniMeditron-Prototype/release_clean/install.sh"  && chmod +x install.sh

## Description 
Minimeditron aims to provide medical field workers with an assistant to patient diagnostic and treatment suggestion.
Minimeditron uses publicly available guidelines to make its judgement leveraging several LLMs to produce its answer.
Minimeditron is able to run locally, offline, on low powered devices for on field usage for humanitarian organisations.

## Theoretical framework

The task realised by minimeditron is split in four tasks, given a case description in human language it performs :
-  structured symptoms identification
-  probabilistic diagnostic propositions
-  potential treatments indications
-  full automatic report generation

### Structured symptoms identification

This subtask consists in producing a list of full symptoms characteristics from a free form text description of a case.
The information taken in account by this task is :
- For each symptom
  - Quantities
  - Qualities
  - Temporality
  - Personal treatment attempted
  - Recent relevant behaviour
  
- Socioeconomic context
- Epidemiological context 
- Physiological context
- Medical history

The output is JSON formatted and condenses only most useful information.

### Probabilistic diagnostic propositions

This subtask takes as input the JSON formatted data of the structured symptoms identification.
It produces a list of possible diagnostics associated with a certainty metric. 
This identification is augmented by classical knowledge tree search in order to compare outputs.
The output is a JSON formatted list of conditions and certainty

### Potential treatments indications

This subtask takes as input the JSON formatted data of the probabilistic diagnostic propositions.
It produces a list of possible treatments associated with a certainty metric and sources.
Used sources are globally recognised guidelines and peer reviewed papers. 
The output is a JSON formatted list of treatments, certainty, document source.

### Full automatic report generation

This subtask takes as input the JSON formatted data of all the other tasks and produces a short report in human-readable
language explaining the logic of the diagnostic and the treatment.
The output is a free form text summarising the whole process.

## Data

### Data sources

The data used is a combination of case studies and guidelines. Each data source must be associated to one or multiple tasks.
Those tasks allow to link a structural representation of the data to the task it is used for.
The tasks are :
- Case description 
  - Comes in three forms : 
    - Contains only symptoms
    - Contains symptoms and diagnosis
    - Contains symptoms, diagnosis and treatment
- Diagnostics : A file containing a list of conditions and their associated symptoms
- Treatments : A file containing a list of treatments and their associated conditions


## Coding framework

In this framework we use two main base level objects :
- Data pipes described by a JSON formatted structure that describe how to process the input
  - Case descriptions are free form texts
  - Symptoms are lists of symptoms (JSON)
  - Diagnostics are lists of conditions (JSON)
  - Treatments are lists of treatments (JSON)
  - Reports are free form text
- Blocks that process a pipe and output a new one
  - Structuriser take free form text and output one of the JSON formatted pipe
  - MultiLabeledClassifierWithCommentary take a JSON formatted pipe and outputs another that represent logits of a classification
  - Verbaliser take a JSON formatted pipe and outputs free form text

### Agents

An agent represents a pipeline of blocks that can be used to process a pipe. Agents have input and output points with descriptions.
We can put multiple agent in a same environment and make the communicate following a protocol so that they can work together in parallel.