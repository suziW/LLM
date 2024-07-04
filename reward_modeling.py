# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/reward_modeling.py \
    --model_name_or_path=facebook/opt-350m \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=16 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --eval_steps=500 \
    --max_length=512 \
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_DISABLED"] = "true"

import warnings
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    GemmaForSequenceClassification,
    OPTForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig

from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config


tqdm.pandas()

model_id = "google/gemma-2b"
model_id = "facebook/opt-350m"

if __name__ == "__main__":
    # parser = HfArgumentParser((RewardConfig, ModelConfig))
    # config, model_config = parser.parse_args_into_dataclasses()
    # config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # print('peft config')
    # print(get_peft_config(model_config))
    # print('quant config')
    # print(get_quantization_config(model_config))
    # exit()

    ################
    # Model & Tokenizer
    ################
    # torch_dtype = (
    #     model_config.torch_dtype
    #     if model_config.torch_dtype in ["auto", None]
    #     else getattr(torch, model_config.torch_dtype)
    # )
    # quantization_config = get_quantization_config(model_config)
    # model_kwargs = dict(
    #     revision=model_config.model_revision,
    #     trust_remote_code=model_config.trust_remote_code,
    #     device_map=get_kbit_device_map() if quantization_config is not None else None,
    #     quantization_config=quantization_config,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_config.model_name_or_path, num_labels=1, **model_kwargs
    # )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        quantization_config=None,  # somehow it put model to gpu, 不quntiza好像model最后几层的tensor都是meta type，内存不够
        # token=hf_token,
        # device_map="auto",
        num_labels=1,
    )

    ################
    # Dataset
    ################
    max_length = 512
    raw_datasets = load_dataset("Anthropic/hh-rlhf")
    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    train_dataset = raw_datasets["train"].select(range(2222))
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=8,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
    )

    eval_dataset = raw_datasets["test"].select(range(22))
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=8,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
    )
    ################
    # Training
    ################
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=2,
        max_steps=100,
        # num_train_epochs=1,
        learning_rate=1.41e-5,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
        optim="adamw_torch",
        evaluation_strategy="steps",
        eval_steps=20,
        # gradient_checkpointing=True,
    )
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,
    )
    trainer.train()
    trainer.save_model("outputs")
    # trainer.push_to_hub()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)
