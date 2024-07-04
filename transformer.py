import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_DISABLED"] = "true"

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
)
from datasets import load_dataset



model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# ----- tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
# print(encoding)
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
print(pt_batch)

# ----- model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
pt_outputs = model(**pt_batch)
print(pt_outputs)
# tokenizer.save_pretrained("test.tokenizer")
# model.save_pretrained("test.model")
# print(model)

# ----- custom config
my_config = AutoConfig.from_pretrained(model_name, n_layers=9)
# print(my_config)


# ----- pipeline
# classifier = pipeline("sentiment-analysis")
# classifier = pipeline("sentiment-analysis", model=model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

result = classifier("fuck u, go away.")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

results = classifier(
    [
        "We are very happy to show you the 🤗 Transformers library.",
        "We hope you don't hate it.",
    ]
)
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


# ----- dataset
dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT
print(dataset)


def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])


dataset = dataset.map(tokenize_dataset, batched=True)
print(dataset)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="outputs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP

trainer.train(resume_from_checkpoint=True)
