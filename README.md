# Persona-Aware Bias Evaluation (PABE)

Evaluating &amp; mitigating bias in chatbot systems through implicit persona encoding

## Motivation

Most modern methods of evaluating bias in LLMs rely on one of two types of methods:

1. examining token probability distribution during masked language modeling of protected attributes
2. examining LLM text generation after token-level perturbation of protected attributes (i.e. explicit addition/modification of token)

These methods are not robust for / not transferrable to evaluating chatbots:

1. chatbots rely on autoregressive (i.e. generative) models, which are not 100% aligned with the MLM task.
2. most in-the-wild chatbot queries never explicitly mention protected attributes in the prompt. do you ever tell explicitly your chatbot you're male?

The Persona-Aware Bias Evaluation (PABE) framework aims to bridge this gap by proposing a new framework to implicitly encode protected attribute information into chatbot queries through zero-shot text style transfer & persona prompting. By injecting a demographic's stylistic voice and persona into a chatbot query, the goal is to simulate a realistic conversation that an individual may have with a chatbot in the wild.

Applications of PABE datasets include evaluation of fairness (i.e. comparing responses from PABE prompts to original reference answers) or mitigation of bias via preferential fine-tuning methods like DPO.

## Data

As a proof-of-concept (PoC) of the PABE framework, we're applying PABE to the iCliniq-10K dataset, which is a collection of real-life patient <> physician conversations in the iCliniq.com online forum. Credits to Li et al. for curating the original dataset (https://github.com/Kent0n-Li/ChatDoctor).

For a 500-row subset of this dataset, we applied the PABE framework to evaluate potential racial biases, focusing on 4 major races/ethnicities in the USA (White, Asian, African American, and Latino/Hispanic). This resulted in a 500\*4 = 2000 row dataset, with the following four features

1. input (string): the original input prompt
2. target (string): the original target (reference) answer/output
3. augmented_prompt (string): PDBD-augmented input prompt
4. persona (string): protected attribute used to augment the persona

To augment the input prompt, we use GPT 3.5 turbo and the OpenAI completions API. Future work will benchmark performance across different open/closed-source models.

## Getting Started

To demo PABE on the iCliniq subset, you need an OpenAI API key. After cloning the repository, create a `.env` file in the root of the repository, and add your API key as `OPENAI_API_KEY`.

Then, make sure you have all the dependencies by creating a virtual environment and installing packages from requirements.txt: `pip install -r requirements.txt`.

Afterwards, run `python augment_dataset.py` to run PABE augmentation. It will download the resulting dataset as a CSV file in /data.

You can run fairness evaluation by running `python evaluate_fairness.py`. Before running, you just need to switch between the GPTDoctor and LlamaDoctor chatbot (refer to the paper) based on which chatbot you want to evaluate.

If you want to finetune LlamaDoctor yourself, the script to do so is available in finetune-chatdoctor.py. However, it requires the HealthCareMagic-100k dataset, which was too large to upload to Git. You can find the online link for it here for manual download (https://drive.google.com/file/d/1lyfqIwlLSClhgrCutWuEe_IACNq6XNUt/view)

# Stylistic Integrity

One major assumption of the PABE framework is that we can encode meaningful and realistic stylistic properties of a demographic's speech via zero-shot text style transfer. To evaluate the stylistic integrity of our dataset's augmented prompts, we turn to methods in LLM-based protected attribute inference suggested by Staab et al. (https://arxiv.org/pdf/2310.07298).

This also relates to the topic of fairness through unawareness, where we assume models that do not explicitly include protected attributes are naturally fair. However, analysis shows that this is not necessarily true: ML models can infer the protected attribute by picking up patterns in correlated features.

By extension, this exercise also tests the fairness through unawareness assumption for an LLM. If an LLM can zero-shot infer protected attributes like race, gender, income, etc. without it explicitly present in a prompt, it speaks volumes about the need for a framework like PABE.

On a high level, we provide an LLM with the role of an expert investigator. Then, we ask the LLM to "profile" the speaker behind a PABE-augmented prompt and provide its top guess (and reasoning) on the speaker's demographic. Then, we compare the predicted protected attribute value with the ground truth value from our dataset to determine how stylistically accurate our generated prompts are. Below are evaluation results across the 4 races/ethnicities we evaluate for this specific dataset.

|             | **Asian** | **African American** | **Latino** | **White** |
| ----------- | --------- | -------------------- | ---------- | --------- |
| **% Match** | _3.8%_    | 71%                  | 63.8%      | 82.8%     |

We note that our LLM (GPT 4o mini) has a difficult time correctly inferring Asian personas from our prompts. This means that PABE fails to generate discriminative prompts unique/distinct to the Asian population. In future work, we will examine the reason behind these failures and explore methods to improve stylistic integrity.

However, for other races/ethnicities, GPT 4o mini does a pretty good job at zero-shot inference. This means that chatbots could, theoretically, indirectly pick up on our protected attributes while chatting with us. We will benchmark stylistic integrity across other protected attributes in future work.
