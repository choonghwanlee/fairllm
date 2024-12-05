from typing import Optional
from collections import defaultdict
from datasets import Dataset
import statsmodels.stats.power as smp
import json
import os
from pydantic import BaseModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def plot_tukey(df: pd.DataFrame, candidate: str):
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        df['mean_diff'],
        df['comparison'],
        xerr=[df['mean_diff'] - df['lower_ci'], df['upper_ci'] - df['mean_diff']],
        fmt='o',
        color='blue'
    )
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Tukey HSD Pairwise Comparisons')
    plt.xlabel('Mean Difference with 95% CI')
    plt.ylabel('Group Comparison')
    plt.savefig(f'{candidate}_tukeyhsd_comparisons.png', dpi=300, bbox_inches='tight')  # Save the plot


def plot_f1boxplot(df: pd.DataFrame, candidate: str):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='persona', y=f'f1_{candidate}', data=df, palette='Set2')
    plt.title('Boxplot of F1 Scores by Race')
    plt.ylabel('F1 Score')
    plt.xlabel('Race')
    plt.savefig('boxplot_f1_scores.png', dpi=300, bbox_inches='tight')  # Save the plot



class PersonaInference(BaseModel):
    guess: str
    reasoning: str

