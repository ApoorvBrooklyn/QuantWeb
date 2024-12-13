# import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# import pandas as pd
# from news_cnbc import processed_news
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


# # Load FinBERT tokenizer and model
# model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model")
# tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")

# # Ensure the model is in evaluation mode
# model.eval()

# # Function for sentiment analysis
# def analyze_sentiment(text):
#     """Analyzes sentiment for a given text using FinBERT."""
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     probabilities = torch.nn.functional.softmax(logits, dim=1)
#     sentiment = torch.argmax(probabilities).item()  # 0: negative, 1: neutral, 2: positive
#     sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
#     return sentiment_map[sentiment], probabilities.numpy()

# # Apply sentiment analysis to the DataFrame
# def add_sentiment_to_dataframe(df):
#     sentiments = []
#     confidences = []
    
#     for headline in df['headline']:
#         sentiment, probs = analyze_sentiment(headline)
#         sentiments.append(sentiment)
#         confidences.append(probs)
    
#     df['sentiment'] = sentiments
#     df['confidence'] = confidences
#     return df

# # Apply to processed_news DataFrame
# processed_news = add_sentiment_to_dataframe(processed_news)

# # Display results
# print(processed_news[['headline', 'sentiment', 'confidence']])

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Expanded rule-based keywords
# rule_based_keywords = {
#     "Negative": [
#         # General terms
#         "war", "dies", "slowdown", "crisis", "depreciation", "weakens", "inflation", "conflict", 
#         "recession", "losses", "defaults", "deficit", "volatility", "unemployment", "sanctions",
#         # Sector-specific terms
#         "market crash", "bear market", "bankruptcy", "supply chain disruption", 
#         "trade restrictions", "currency depreciation", "rate hike",
#     ],
#     "Neutral": [
#         # General terms
#         "mixed effects", "volatility", "cautious optimism", "stagnation", "temporary recovery",
#         "stability", "no significant change",
#         # Sector-specific terms
#         "trade balance", "moderate growth", "flat market", "unchanged interest rates", 
#         "fiscal neutrality", "steady demand",
#     ],
#     "Positive": [
#         # General terms
#         "growth", "boost", "strengthens", "record profits", "surplus", "recovery", "stabilizes",
#         "expansion", "bull market", "job creation", "exports rise", "foreign investments",
#         # Sector-specific terms
#         "economic recovery", "capital inflow", "surplus growth", "rate cut", 
#         "trade deal", "innovation breakthrough", "currency appreciation", 
#         "market rally", "policy reform",
#     ],
# }
rule_based_keywords = {
    "Negative": [
        # General negative terms
        "war", "dies", "slowdown", "crisis", "recession", "inflation", "volatility", 
        "depreciation", "defaults", "losses", "conflict", "unemployment", "deficit", 
        "stagnation", "plummets", "plunge", "decline", "drops", "surge in costs",
        # Economic and financial terms
        "market crash", "bear market", "currency depreciation", "rate hike", 
        "economic contraction", "fiscal deficit", "negative growth", 
        "capital outflow", "bond yield spikes", "credit crunch", 
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
        "currency appreciation", "market rally", "capital inflow", "economic boom",
        # Economic and financial terms
        "fiscal surplus", "rate cut", "economic recovery", "rebound", 
        "strong earnings", "positive outlook", "GDP growth", "investment surge", 
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

# Test sentences
test_sentences = [
    "India faces war situation",
    "Prime Minister of India dies of internal conflict",
    "India's forex reserve at its highest",
    "Rupee weakens drastically against Dollar",
    "Economic slowdown in India",
    "Indian markets rally on global capital inflows",
    "US trade restrictions disrupt global supply chains",
    "Bull market observed in European technology sector",
    "Record profits reported by Indian IT firms",
    "Unemployment rates surge due to economic slowdown",
]

# Run the enhanced sentiment analysis
analysis_results = sentiment_analysis(test_sentences)

# Display results
for result in analysis_results:
    print(f"Sentence: {result['sentence']}")
    print(f"Predicted Sentiment: {result['sentiment']}")
    print()
