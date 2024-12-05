from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dotenv import load_dotenv
import torch
import os

from utils import plot_tukey, plot_f1boxplot
from inference import inference_gpt4, inference_chatdoctor
from datasets import load_dataset
from openai import OpenAI



def evaluate_STS(df: pd.DataFrame, candidate: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    embeddings1 = model.encode(df[candidate + '_response'], convert_to_tensor=True)
    embeddings2 = model.encode(df["target"], convert_to_tensor=True)

    similarities = util.cos_sim(embeddings1, embeddings2).diagonal()
    df[f'sts_{candidate}'] = similarities.cpu().tolist()
    groups = [df[df['persona'] == g][f'sts_{candidate}'] for g in df['persona'].unique()]
    anova_result = f_oneway(*groups)
    print(f"ANOVA p-value: {anova_result.pvalue}")
    tukey = pairwise_tukeyhsd(endog=df[f'sts_{candidate}'], groups=df['persona'], alpha=0.05)
    tukey_summary = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_summary['lower_ci'] = tukey_summary['lower'].astype(float)
    tukey_summary['upper_ci'] = tukey_summary['upper'].astype(float)
    tukey_summary['mean_diff'] = tukey_summary['meandiff'].astype(float)
    tukey_summary['comparison'] = tukey_summary['group1'] + ' vs. ' + tukey_summary['group2']
    return df, tukey_summary


if __name__ == '__main__':
    # load dataset
    dataset = load_dataset('jasonhwan/iCliniq500-pdbd-race', 'tst', split='train')
    ## conduct inference
    print('beginning inference...')
    # dataset = dataset.map(inference_gpt4, num_proc=os.cpu_count()-1) ## switch to GPTDoctor inference
    dataset = dataset.map(inference_chatdoctor, batched=True, batch_size=8) ## switch to LlamaDoctor inference
    df = dataset.to_pandas()
    ## evaluate fairness
    print('beginning evaluation...')
    df, tukey_summary = evaluate_STS(df, 'chatdoctor') ## switch between chatdoctor and gpt4 based on which chatbot you're evaluating
    plot_tukey(tukey_summary, 'chatdoctor')
    plot_f1boxplot(df, 'chatdoctor')
    print('done!')



