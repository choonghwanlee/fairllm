from prompts import pii_inference_template
from utils import PersonaInference, power_analysis
from openai import OpenAI
import re
import os 
from dotenv import load_dotenv
from datasets import load_dataset


def evaluate_personas(example: dict, attributes: str):
    '''
    Evaluate whether LLM-inferred persona matches ground truth persona

    Input:
        example (dictionary): a row from our dataset
        client: an OpenAI API compatible model
        attribute (str): comma-separated list of protected attributes we want to infer 
    '''
    load_dotenv()
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages= [
            {"role": "system", "content": pii_inference_template.format(attributes=attributes)},
            {"role": "user", "content": example['augmented_prompt']}
        ],
        temperature = 0,
        max_tokens=512,
    )
    response = completion.choices[0].message.content
    matches = re.findall(r'\[(.*?)\]', response) ## regex matching of output
    if matches[0] == example['persona']:
        return {"matched": 1} 
    return {"matched": 0}

if __name__ == "__main__": 
    ## fetch api key through argparser
    dataset = load_dataset('jasonhwan/iCliniq500-pdbd-race', 'tst', split='train')
    ## Power Analysis
    power = power_analysis(0.8, 0.05, len(dataset.unique('persona')), len(dataset))
    print("Cohen's f^2 for the dataset: ", power)
    ## Persona Inference – speed up by scaling CPU usage
    dataset = dataset.map(evaluate_personas, fn_kwargs={'attributes': ", ".join(dataset.unique('persona'))}, num_proc=os.cpu_count()-1)
    ## Analysis of success rate across demographics
    df = dataset.to_pandas()
    success_rates = df.groupby("persona")["matched"].mean()
    # Print success rates
    for group, rate in success_rates.items():
        print(f"Success rate for {group}: {rate:.2%}")