# helper functions for precision metrics

import json
import re
import os
from fuzzywuzzy import fuzz
import pandas as pd
import matplotlib.pyplot as plt

# PRECISION METRIC

def extract_conditions(json_file):
    '''
    Extracts the conditions from the json file 
    '''
    parsed_json = json.loads(json_file)
    extracted_conditions = parsed_json["output_structure"]["Names of Conditions"]
    return extracted_conditions

def exact_matching(ground_truth, extracted_conditions):
    '''
    Returns the positions of the exact matches between the ground truth and the extracted conditions
    '''
    matching_positions = []
    for idx, condition in enumerate(extracted_conditions):
        if condition == ground_truth:
            matching_positions.append(idx)
    return matching_positions

def fuzzy_matching(ground_truth, extracted_conditions, threshold=80):
    '''
    Returns the positions of the fuzzy matches between the ground truth and the extracted conditions
    '''
    matching_positions = []
    fuzzy_scores = []
    for idx, condition in enumerate(extracted_conditions):
            score = fuzz.ratio(condition, ground_truth)
            fuzzy_scores.append(score)
            if score >= threshold:
                matching_positions.append((idx, score))

    return fuzzy_scores, matching_positions

# HANDCRAFTED METRICS

def load_and_retreive(json_file):
    '''
    Load and retreive the conditon and related conditions from a well formated json file
    '''
    with open(json_file) as f:
        answer = json.load(f)
        condition = answer['document_structure']['Condition']
        related_conditions = answer['document_structure']['Related diagnosis']
        f.close()
    return condition, related_conditions

def retreive_from_string(prompt):
    '''
    Retreive condition and related conditions from prompt string
    '''
    condition_match = re.search(r'"Condition":\s*"([^"]+)"', prompt)
    condition = condition_match.group(1) if condition_match else None
    related_conditions_match = re.search(r'"Related diagnosis":\s*\[([^\]]+)\]', prompt)
    related_conditions = [s.strip(' "') for s in related_conditions_match.group(1).split(',')] if related_conditions_match else None
    return condition, related_conditions

def single_fuzzy_matching(ground_truth, condition):
    '''
    Fuzzy matching
    '''
    score = fuzz.ratio(condition, ground_truth)
    return score

def classify_condition(condition, ground_truth, related_conditions, THRESHOLD):
    ''' 
    Classify the condition into 3 categories : correct, rekated, unrelated
    '''
    if condition == ground_truth or single_fuzzy_matching(condition, ground_truth) > THRESHOLD:
        return "correct"
    
    for cond in related_conditions:
        if exact_matching(cond, ground_truth) or single_fuzzy_matching(cond, ground_truth) > THRESHOLD:
            return "related"
    
    return "unrelated"


# embeddings metrics

import torch
import torch.nn.functional as F

def tensor_distance(tensor1, tensor2, distance_type="L2"):
    """
    Compute the L2 (Euclidean) distance between two tensors of the same shape.
    
    Args:
    - tensor1 (torch.Tensor): The first tensor.
    - tensor2 (torch.Tensor): The second tensor.
    - distance_type (str): The type of distance to compute. Currently only
    
    Returns:
    - float: The distance / similarity between the two tensors.
    """
    
    if tensor1.shape != tensor2.shape:
        raise ValueError("Both tensors must have the same shape.")

    if distance_type == "L2":
        distance = torch.norm(tensor1 - tensor2)
    elif distance_type == "Manhattan":
        distance = torch.sum(torch.abs(tensor1 - tensor2))
    elif distance_type == "Cosine":
        similarity = F.cosine_similarity(tensor1, tensor2)
        distance = 1 - similarity
    elif distance_type == "Minkowski":
        distance = torch.norm(tensor1 - tensor2, p=3)
    
    return distance