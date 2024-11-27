## Use the OpenAI API to augment each row of the dataset
from openai import OpenAI
import json
import numpy as np 
from prompts import augment_with_system, augment_with_tts
from dotenv import load_dotenv
from datasets import load_dataset
from utils import clean_icliniq
import argparse



def apply_with_system(example, client, persona_list: list[str]):
    '''
    Role prompting via system prompts

    Did not use in practice, but would be cool to benchmark in the future

    Input:
        example (dictionary): a batch of rows from our dataset 
        client: an OpenAI API compatible model to use for persona augmentation
        prompt (string): the input prompt format
        persona (string): the persona we want the user to have
    Returns:
        example (dictionary): updated rows from our dataset
    '''
    completions, inputs, targets = [], [], []
    for persona in persona_list:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages= [
                {"role": "system", "content": augment_with_system.format(persona=persona)},
                {"role": "user", "content": example['input'][0]}
            ],
            max_completion_tokens = 512,
            temperature = 0,
        ) 
        completion = response.choices[0].message.content
        completions.append(completion)
        inputs.append(example['input'][0])
        targets.append(example['answer_icliniq'][0])
    return {'input': inputs, 'target': targets, 'augmented_prompt': completions, 'persona': persona_list}

def apply_with_completion(example: dict, client, persona_list: list[str]):
    '''
    Zero-shot text style transfer via Completion API 

    Input:
        example (dictionary): a batch of rows from our dataset 
        client: an OpenAI API compatible model to use for persona augmentation
        user_prompt (string): the input prompt format
        persona (string): the persona we want the user to have
    Returns:
        example (dictionary): updated rows from our dataset
    '''
    completions, inputs, targets = [], [], []
    for persona in persona_list:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt = augment_with_tts.format(prompt = example['input'][0], persona=persona),
            max_tokens = 512,
            temperature = 0,
        ) 
        completion = response.choices[0].text
        completions.append(completion)
        inputs.append(example['input'][0])
        targets.append(example['answer_icliniq'][0])
    return {'input': inputs, 'target': targets, 'augmented_prompt': completions, 'persona': persona_list}


if __name__ == "__main__": 
    ## fetch api key through argparser
    load_dotenv()
    client = OpenAI() ## add API key
    with open('protected_attributes.json', 'r') as file:
        attributes = json.load(file)
    race_attributes = attributes['race']
    ## load iCliniq dataset
    dataset = load_dataset("json", data_files='./data/iCliniq.json')
    ## data cleaning
    cleaned_icliniq = clean_icliniq(dataset['train'])
    ## select smaller subset
    icliniq_500subset = cleaned_icliniq.shuffle(seed=42).select(range(500))
    ## PABE data augmentation via Completion
    system_icliniq = icliniq_500subset.map(apply_with_system, fn_kwargs={'client': client, 'persona_list': race_attributes}, 
                                             batched=True, batch_size=1, remove_columns=['input', 'answer_icliniq', 'answer_chatgpt', 'answer_chatdoctor'])
    ## export results 
    system_icliniq.to_csv('./data/iCliniq500-pbdb.csv')
    # system_icliniq.push_to_hub("jasonhwan/iCliniq500-pdbd-race", "persona")
