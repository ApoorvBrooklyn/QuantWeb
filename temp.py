from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")

# Test sentences
test_sentences = [
   ""
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
