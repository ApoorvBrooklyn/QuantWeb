from transformers import BertTokenizer, BertForSequenceClassification
import torch
#from news_cnbc import processed_news

# Load the FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
import pandas as pd

# Assuming 'processed_news' is a DataFrame
#headlines = processed_news.iloc[:, 0].tolist()  # Replace 0 with the actual column index for headlines if needed


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
    return None  # If no keyword matches, return None

# Function to classify sentiment (rule-based + model-based)
def sentiment_analysis(sentences):
    results = []
    for sentence in sentences:
        # Check for rule-based classification
        rule_sentiment = classify_sentiment_with_rules(sentence)
        if rule_sentiment:
            results.append({"sentence": sentence, "sentiment": rule_sentiment})
        else:
            # If no rule applies, use the model
            inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            # Map numerical labels back to sentiments
            label_map_reverse = {0: "Negative", 1: "Neutral", 2: "Positive"}
            model_sentiment = label_map_reverse[prediction]
            results.append({"sentence": sentence, "sentiment": model_sentiment})
    return results

test_sentences = ["Us cuts fed rates"]

# Run the enhanced sentiment analysis
analysis_results = sentiment_analysis(test_sentences)

# Display results
for result in analysis_results:
    print(f"Sentence: {result['sentence']}")
    print(f"Predicted Sentiment: {result['sentiment']}")
    print()
