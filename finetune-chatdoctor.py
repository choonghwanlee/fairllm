import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from prompts import chatdoctor_template
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset, load_dataset


## Data Loading
dataset = load_dataset('json', './data/HealthCareMagic-100k.json')
healthcaremagic_subset = dataset['train'].shuffle(seed=42).select(range(50000))


max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)


model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state = 32,
    loftq_config = None,
)
print(model.print_trainable_parameters())

EOS_TOKEN = tokenizer.eos_token
def formatting_prompt(examples):
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for input_, output in zip(inputs, outputs):
        text = chatdoctor_template.format(input_, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


healthcaremagic_subset = healthcaremagic_subset.map(formatting_prompt, batched=True)

trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=healthcaremagic_subset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

trainer.train()

model.push_to_hub("jasonhwan/chatdoctor_llama3.2_3B_4bit") ## replace with your repo
tokenizer.push_to_hub("jasonhwan/chatdoctor_llama3.2_3B_4bit") ## replace with your repo