from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch


# Load the extended dataset
data_file = "extended_financial_sentiment_dataset.csv"
data = pd.read_csv(data_file)

# Convert the dataset to Hugging Face format
hf_dataset = Dataset.from_pandas(data)

# Label mapping
label_mapping = {"Positive": 2, "Neutral": 1, "Negative": 0}

def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("./fintwitbert_sentiment_finetuned")
    inputs = tokenizer(examples["News_Headline"], truncation=True, padding=True, max_length=128)
    inputs["labels"] = [label_mapping[label] for label in examples["Sentiment"]]
    return inputs

# Apply preprocessing
hf_dataset = hf_dataset.map(preprocess_function, batched=True)
hf_dataset = hf_dataset.train_test_split(test_size=0.2)

# Load the already fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("./fintwitbert_sentiment_finetuned", num_labels=3)

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./fintwitbert_sentiment_continued",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# Define compute_metrics function
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1).cpu().numpy()
    labels = torch.tensor(labels).cpu().numpy()
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset["train"],
    eval_dataset=hf_dataset["test"],
    tokenizer=AutoTokenizer.from_pretrained("./fintwitbert_sentiment_finetuned"),
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the continued fine-tuned model
trainer.save_model("./fintwitbert_sentiment_continued")

print("Continued fine-tuning completed and model saved.")