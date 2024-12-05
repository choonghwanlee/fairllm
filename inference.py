from prompts import chatgpt_doctor
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
# from unsloth import FastLanguageModel
from prompts import chatdoctor_template
from openai import OpenAI
from dotenv import load_dotenv
import torch

def inference_gpt4(example: dict):
    ## load model
    load_dotenv()
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages= [
            {"role": "system", "content": chatgpt_doctor},
            {"role": "user", "content": example['augmented_prompt']}
        ],
        max_completion_tokens = 512,
        temperature = 0,
    ) 
    completion = response.choices[0].message.content
    example['gpt4_response'] = completion
    return example


def inference_chatdoctor(examples: list[dict], model_id: str):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # if torch.cuda.is_available(): model = FastLanguageModel.for_inference(model)
    inputs = tokenizer(
    [
        chatdoctor_template.format(
            #instructions
            example,
            #answer
            "",
        ) for example in examples['augmented_prompt']
    ], return_tensors = "pt", padding=True).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
    answers = tokenizer.batch_decode(outputs)
    answers = [answer.split("### Response:")[-1] for answer in answers]
    examples['chatdoctor_response'] = answers
    return examples






