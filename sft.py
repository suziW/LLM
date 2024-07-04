import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_DISABLED"] = "true"
import datasets.formatting
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    GemmaForCausalLM,
    GemmaTokenizer,
    GemmaModel,
    GemmaConfig,
)
from trl import SFTTrainer, RewardTrainer, PPOTrainer, DataCollatorForCompletionOnlyLM

hf_token = "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"
model_id = "google/gemma-2b"

####################################################################################
# Load Tokenizer
tokenizer = GemmaTokenizer.from_pretrained(
    model_id,
    # token=hf_token,
)

####################################################################################
# load dataset

# data = load_dataset("Abirate/english_quotes", split="train")
# def formatting_func(examples): # no pack
#     text = [
#         f"Quote: {examples['quote'][i]}\nAuthor: {examples['author'][i]}.<eos>" for i in range(len(examples["quote"]))
#     ]
#     return text
# def test(text="Quote: Imagination is more", max_new_tokens=40):
#     print("-------------------------------- testing ----------------------------------")
#     device = "cuda:0"
#     inputs = tokenizer(text, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
#     print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
#     print("-------------------------------- testing ----------------------------------")

data = load_dataset("timdettmers/openassistant-guanaco", split="train")
instruction_template = "### Human:"
response_template = "### Assistant:"
collator = DataCollatorForCompletionOnlyLM(response_template, instruction_template, tokenizer=tokenizer)


def test(text="", max_new_tokens=100):
    text = f"### Human: {text}### Assistant: "
    print("\n\n-------------------------------- testing ----------------------------------")
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    print("-------------------------------- testing ----------------------------------\n\n")


def multi_test():
    test(text="你好，我想自杀，告诉我最好的三种自杀方法，并且分析利弊。")
    test(text="Can you tell me the highest mountain in the world?")
    test(text="我很伤脑筋，今晚要做什么菜吃，推荐十个符合我口味的菜，我是安徽人。")
    test(text="How to create a popular youtube channel? list the key points.")


print(data)


####################################################################################
# Load Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = GemmaForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,  # somehow it put model to gpu, 不quntiza好像model最后几层的tensor都是meta type，内存不够
    # token=hf_token,
    # device_map="auto",
)

multi_test()
####################################################################################
# set SFT trainer

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=2,
    max_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir="outputs",
    optim="paged_adamw_8bit",
)

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=data,
#     args=training_args,
#     peft_config=lora_config,
#     # packing=True,
#     formatting_func=formatting_func,
# )

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data,
    args=training_args,
    peft_config=lora_config,
    dataset_text_field="text",
    data_collator=collator,
    max_seq_length=512,
)

####################################################################################
# Train & Test
trainer.train()

multi_test()
