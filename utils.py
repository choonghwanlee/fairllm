from typing import Optional
from collections import defaultdict
from datasets import Dataset
import statsmodels.stats.power as smp
import json
import os
from pydantic import BaseModel

def clean_icliniq(dataset: Dataset):
    '''
    Clean iCliniq dataset by removing incomplete & short responses + filtering for shorter prompts

    Input:
        dataset (Dataset): a Dataset object from the HuggingFace Datasets library
    '''

    ## remove incomplete sentence
    cleaned_dataset = dataset.filter(lambda example: example['answer_icliniq'].rstrip().endswith('.'))
    
    ## remove rows with target response that is too short (i.e. responses with insufficient information, usually formalities & basic responses with no diagnostic depth)
    cleaned_dataset = cleaned_dataset.filter(lambda example: len(example['answer_icliniq']) >= 225)

    ## only include rows with prompts that are of short-to-moderate length (i.e. remove outliers in the tail end of length distribution)
    cleaned_dataset = cleaned_dataset.filter(lambda example: len(example['input']) <= 875)

    return cleaned_dataset

def power_analysis(power: float, alpha: float, num_classes: int, n: int):
    '''
    Power Analysis with Cohen's f2 for the multiclass case

    Input:
        power (float): desired power of dataset
        alpha (float): desired alpha value for dataset
        num_classes (int): number of distinct classes in dataset
        n (int): number of prompt <> response pairs in the dataset 
    '''
    # Perform power analysis to calculate the minimum detectable effect size (Cohen's f²)
    analysis = smp.FTestAnovaPower()
    effect_size = analysis.solve_power(effect_size=None, nobs=n, alpha=alpha, power=power, k_groups=num_classes)
    return effect_size

class PersonaInference(BaseModel):
    guess: str
    reasoning: str

