import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GemmaForCausalLM
from peft import LoraConfig
import torch
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, setup_chat_format
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
# instruction_template = "### Human:"
# response_template = "### Assistant:"
# collator = DataCollatorForCompletionOnlyLM(response_template, instruction_template, tokenizer=tokenizer)

dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")

# Load Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = GemmaForCausalLM.from_pretrained(
    "google/gemma-2b",
    quantization_config=bnb_config,
    # token=hf_token,
    # device_map="auto",
)
setup_chat_format(model, tokenizer)



lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    # packing=True,
    peft_config=lora_config,
)

trainer.train()
