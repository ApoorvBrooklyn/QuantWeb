import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from news_cnbc import processed_news

# Load FinBERT tokenizer and model
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Function for sentiment analysis
def analyze_sentiment(text):
    """Analyzes sentiment for a given text using FinBERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    sentiment = torch.argmax(probabilities).item()  # 0: negative, 1: neutral, 2: positive
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_map[sentiment], probabilities.numpy()

# Apply sentiment analysis to the DataFrame
def add_sentiment_to_dataframe(df):
    sentiments = []
    confidences = []
    
    for headline in df['headline']:
        sentiment, probs = analyze_sentiment(headline)
        sentiments.append(sentiment)
        confidences.append(probs)
    
    df['sentiment'] = sentiments
    df['confidence'] = confidences
    return df

# Apply to processed_news DataFrame
processed_news = add_sentiment_to_dataframe(processed_news)

# Display results
print(processed_news[['headline', 'sentiment', 'confidence']])