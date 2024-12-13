from transformers import BertTokenizer, BertForSequenceClassification
import torch
import openai
import pandas as pd
from dotenv import load_dotenv
import os
from news_cnbc import processed_news  # Assuming this is your source of data

# Load .env file for OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Assuming 'processed_news' is a DataFrame
headlines = processed_news.iloc[:, 0].tolist()

# Rule-based keywords for sentiment classification
rule_based_keywords = {
    "Negative": [
        # General negative terms
        "war", "dies", "slowdown", "crisis", "recession", "inflation", "volatility", 
        "depreciation", "defaults", "losses", "conflict", "unemployment", "deficit", 
        "stagnation", "plummets", "plunge", "decline", "drops", "surge in costs",
        # Economic and financial terms
        "market crash", "bear market", "currency depreciation", "rate hike", 
        "economic contraction", "fiscal deficit", "negative growth", 
        "capital outflow", "bond yield spikes", "credit crunch", "shrink", 
        # Trade and international relations
        "supply chain disruption", "trade restrictions", "sanctions", "tariff increase", 
        "trade war", "protectionism", "export restrictions",
        # Political and environmental
        "political instability", "natural disaster", "terrorist attack", "regulatory hurdles",
        "climate crisis", "environmental disaster", "civil unrest", "protest",
        # Industry-specific
        "factory shutdown", "job cuts", "bankruptcy", "layoffs", "demand slump", 
        "energy shortage", "commodity price hike", "oil prices soar", 
    ],
    "Neutral": [
        # General neutral terms
        "mixed effects", "volatility", "stagnation", "moderate growth", "unchanged", 
        "steady demand", "flat market", "no significant change", "temporary impact",
        # Economic and financial terms
        "stable currency", "balanced budget", "trade balance", "moderate inflation", 
        "status quo", "neutral stance", "steady recovery", 
        # Trade and international relations
        "new trade policies", "bilateral agreements", "steady exports", "gradual improvement",
        # Political and environmental
        "policy discussions", "coalition talks", "temporary disruptions", 
        # Industry-specific
        "mixed sectoral growth", "slow-paced innovation", "gradual adoption",
    ],
    "Positive": [
        # General positive terms
        "growth", "boost", "strengthens", "stabilizes", "record profits", "recovery", 
        "surplus", "expansion", "bull market", "exports rise", "job creation", 
        "currency appreciation", "market rally", "capital inflow", "economic boom", "tax cuts"
        # Economic and financial terms
        "fiscal surplus", "rate cut", "economic recovery", "rebound", 
        "strong earnings", "positive outlook", "GDP growth", "investment surge", "cut rates"
        # Trade and international relations
        "trade deal", "exports boost", "reduced tariffs", "global cooperation",
        "new partnerships", "strengthened alliances", 
        # Political and environmental
        "policy reform", "regulatory relief", "peace talks", "environmental breakthroughs", 
        "technological innovation",
        # Industry-specific
        "record production", "increased demand", "strong sales", "supply chain efficiency",
        "renewable energy growth", "tech breakthroughs", "new product launches",
    ],
}

# Function to classify sentiment based on rules
def classify_sentiment_with_rules(sentence):
    sentence_lower = sentence.lower()
    for sentiment, keywords in rule_based_keywords.items():
        if any(keyword in sentence_lower for keyword in keywords):
            return sentiment
    return None

# Function to classify sentiment using OpenAI API
def classify_sentiment_openai(sentence):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial sentiment analysis expert."},
                {"role": "user", "content": f"Classify the sentiment of this sentence as Positive, Neutral, or Negative based on its financial impact on the Indian Stock Market: '{sentence}'"}
            ],
            max_tokens=10,
            temperature=0
        )
        sentiment = response['choices'][0]['message']['content'].strip()
        return sentiment
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "Error"

# Function to classify sentiment (rule-based, model-based, and OpenAI API)
def sentiment_analysis(sentences):
    results = []
    for sentence in sentences:
        rule_sentiment = classify_sentiment_with_rules(sentence)
        if rule_sentiment:
            results.append({"sentence": sentence, "sentiment": rule_sentiment, "source": "Rule-Based"})
        else:
            # FinBERT model-based classification
            inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            label_map_reverse = {0: "Negative", 1: "Neutral", 2: "Positive"}
            model_sentiment = label_map_reverse[prediction]

            # Validate with OpenAI API
            openai_sentiment = classify_sentiment_openai(sentence)

            # Combine or flag discrepancies
            if model_sentiment != openai_sentiment:
                results.append({
                    "sentence": sentence,
                    "sentiment": model_sentiment,
                    "openai_sentiment": openai_sentiment,
                    "source": "FinBERT + OpenAI (Discrepancy)"
                })
            else:
                results.append({
                    "sentence": sentence,
                    "sentiment": model_sentiment,
                    "source": "FinBERT"
                })
    return results

# Run the enhanced sentiment analysis
analysis_results = sentiment_analysis(headlines)

# Save results to a CSV file for further analysis
results_df = pd.DataFrame(analysis_results)
results_df.to_csv("sentiment_analysis_results.csv", index=False)

# Display results
for result in analysis_results:
    print(f"Sentence: {result['sentence']}")
    print(f"Predicted Sentiment: {result['sentiment']}")
    if "openai_sentiment" in result:
        print(f"OpenAI Sentiment: {result['openai_sentiment']}")
    print(f"Source: {result['source']}")
    print()
