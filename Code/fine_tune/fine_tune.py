from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import pandas as pd
import datasets
from datasets import load_dataset

from transformers import PreTrainedTokenizer
from typing import Dict
import numpy as np

import os
import torch
import torch.distributed as dist


# Initialize distributed training
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'  # Choose a free port number
# dist.init_process_group(backend='nccl')

modelName = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    modelName, device_map="auto", cache_dir="/home/hice1/ckniffin6/scratch/DL/project")
tokenizer = AutoTokenizer.from_pretrained(modelName)

base_prompt_template: str = """
    You are a financial analyst designed to answer questions about business and finance. Your job is to reply to questions about finance topics and provide advice.

    Instruction: {query}
    
    Output:
"""


# Data loading
base_dataset = load_dataset("gbharti/finance-alpaca", split="train").to_pandas()
base_dataset["input"] = base_dataset["instruction"] + base_dataset["input"]
dataset = base_dataset.drop(["instruction", "text"], axis=1)


def template(inputText):
    return base_prompt_template.format(query=inputText)


dataset["input"] = dataset["input"].apply(template)
tokenizer.pad_token = tokenizer.eos_token
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)


def tokenize_input_output(example):
    return tokenizer(example["input"], example["output"], padding="max_length", return_tensors="pt", truncation=True, max_length=550)


dataset = datasets.Dataset.from_pandas(dataset).map(tokenize_input_output, batched=True)
dataset = dataset.train_test_split(test_size=0.2)


# Training Setup
model.train()
model.gradient_checkpointing_enable()
# model.print_trainable_parameters()


# hyperparameters
lr = 2e-4
batch_size = 10
num_epochs = 3

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir="training",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    # may need to comment out this line
    # fsdp='full_shard',
    # local_rank=int(os.environ['LOCAL_RANK'])
)


# configure trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    data_collator=data_collator
)

# Training Call
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
model.config.use_cache = True
trainer.save_model("/home/hice1/ckniffin6/scratch/DL/project/fine_tune")


# Final Check
model.eval()
query = "What's the difference between a stock and an option?"
query = base_prompt_template.format(query=query)
modelInput = tokenizer(query, return_tensors="pt").to("cuda")
outputs = model.generate(**modelInput, max_new_tokens=500, do_sample=True)
print(tokenizer.batch_decode(outputs)[0])
