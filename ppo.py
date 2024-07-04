import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_DISABLED"] = "true"

from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, DistilBertModel

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available


tqdm.pandas()

hf_token = "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"
model_name = "google/gemma-2b"
model_name = "lvwerra/gpt2-imdb"
dataset_name = "imdb"

####################################################################################
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if model_name == "lvwerra/gpt2-imdb":
    tokenizer.pad_token = tokenizer.eos_token


####################################################################################
# Prepare dataset
def build_dataset(query_dataset="imdb", input_min_text_length=2, input_max_text_length=8):
    # load imdb with datasets
    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


dataset = build_dataset(query_dataset=dataset_name)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


####################################################################################
# Load Reward Model
sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb")
text = "this movie was really bad!!"
score = sentiment_pipe(text, top_k=None, function_to_apply="none")
print(score)


####################################################################################
# Load SFT Model
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    # peft_config=peft_config,
)
ref_model = None


####################################################################################
# Initialize PPOTrainer
config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=128,
    mini_batch_size=1,
    gradient_accumulation_steps=16,
)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

####################################################################################
# Training loop
for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, top_k=None, function_to_apply="none")
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
    ref_pipe_outputs = sentiment_pipe(ref_texts, top_k=None, function_to_apply="none")
    ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]
    batch["ref_rewards"] = ref_rewards

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
