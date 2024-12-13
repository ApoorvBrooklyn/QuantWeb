from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch

# === Load and Prepare Dataset ===
# Load synthetic dataset (first 2000 entries)
df = pd.read_csv("cleaned_dataset.csv").head(2000)
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

# === Training the Model ===
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

# === Load and Test the Model ===
# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")

# Test sentences
test_sentences = [
    "The company reported strong quarterly earnings.",
    "The market is uncertain about the upcoming elections.",
    "The project faced significant delays, leading to losses."
]

# Tokenize the test sentences
inputs = tokenizer(test_sentences, padding=True, truncation=True, return_tensors="pt")

# Run the model on the test sentences
with torch.no_grad():
    outputs = model(**inputs)

# Extract logits and convert to predictions
logits = outputs.logits
predictions = torch.argmax(logits, dim=1)

# Map numerical labels back to sentiments
label_map_reverse = {0: "Negative", 1: "Neutral", 2: "Positive"}
predicted_labels = [label_map_reverse[label.item()] for label in predictions]

# Display results
for sentence, prediction in zip(test_sentences, predicted_labels):
    print(f"Sentence: {sentence}")
    print(f"Predicted Sentiment: {prediction}")
    print()
