from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load synthetic dataset
df = pd.read_csv("cleaned_dataset.csv")
dataset = Dataset.from_pandas(df)

# Map sentiment to numerical labels
label_map = {"Positive": 2, "Neutral": 1, "Negative": 0}
dataset = dataset.map(lambda x: {"labels": label_map[x["sentiment"]]})

# Tokenizer and model
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch["headline"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()
