import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

class EnhancedSentimentAnalyzer:
    def __init__(self, model_name="StephanAkkerman/FinTwitBERT-sentiment"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.rule_based_keywords = {
             "Negative": [
        # Economic Downturn
        "recession", "depression", "economic collapse", "market crash", "financial crisis", 
        "economic meltdown", "economic downturn", "stagflation", "economic contraction",
        
        # Financial Performance
        "loss", "losses", "deficit", "debt", "bankruptcy", "insolvency", "default", 
        "financial strain", "underwater", "write-down", "negative earnings", "margin squeeze",
        
        # Market Indicators
        "bear market", "market decline", "stock plunge", "market correction", "selloff", 
        "market volatility", "market tumble", "market downturn", "market panic",
        
        # Job Market
        "layoffs", "job cuts", "workforce reduction", "unemployment", "hiring freeze", 
        "redundancies", "mass layoffs", "terminated", "downsizing",
        
        # Monetary Challenges
        "inflation", "hyperinflation", "currency devaluation", "currency collapse", 
        "economic sanctions", "trade war", "capital flight", "credit crunch",
        
        # Business Challenges
        "supply chain disruption", "production halt", "factory closure", "business failure", 
        "revenue drop", "profit warning", "negative guidance", "cost overrun",
        
        # Investment Risks
        "market risk", "investment loss", "portfolio decline", "asset depreciation", 
        "investment write-off", "negative return", "underwater investment",
        
        # Negative Financial Terms
        "downgrade", "negative outlook", "credit rating cut", "sector decline", 
        "regulatory penalty", "financial restriction", "economic sanction",
        
        # Negative Performance Words
        "plummet", "crash", "collapse", "decline", "shrink", "drop", "fall", "tumble", 
        "nosedive", "slump", "deteriorate", "worsen", "erode", "underperform",
        
        # Specific Financial Challenges
        "trade deficit", "budget deficit", "negative trade balance", "capital loss", 
        "negative cash flow", "debt spiral", "margin call", "underwater mortgage",
        
        # Global Economic Challenges
        "global recession", "economic slowdown", "emerging market crisis", 
        "international trade conflict", "geopolitical tension", "commodity price crash",
        
        # Industry-Specific Negative Terms
        "tech sector downturn", "retail apocalypse", "energy sector collapse", 
        "real estate market crash", "banking crisis", "automotive industry decline",
        
        # Risk and Uncertainty
        "market uncertainty", "economic instability", "political risk", "regulatory risk", 
        "systemic risk", "volatility", "unpredictability", "market turbulence", "Economic crisis"
    ],
    
    "Neutral": [
        # Economic Stability
        "steady", "stable", "unchanged", "consistent", "moderate", "balanced", 
        "sustainable", "gradual", "measured", "neutral",
        
        # Market Conditions
        "flat market", "sideways market", "range-bound", "consolidation", 
        "no significant change", "status quo", "market equilibrium",
        
        # Financial Performance
        "break-even", "neutral performance", "maintenance mode", "steady state", 
        "consistent performance", "neither growing nor declining",
        
        # Policy and Regulation
        "neutral policy", "balanced approach", "moderate regulation", "wait and see", 
        "under review", "pending decision", "neutral stance",
        
        # Economic Indicators
        "unchanged rates", "stable inflation", "consistent growth", "moderate expansion", 
        "steady employment", "predictable market", "normalized conditions",
        
        # Investment Terms
        "neutral outlook", "balanced portfolio", "diversified investment", 
        "risk-neutral strategy", "hedged position", "balanced exposure",
        
        # Trade and Commerce
        "trade balance", "neutral trade", "consistent imports", "steady exports", 
        "moderate international relations", "balanced trade",
        
        # Generalized Neutral Terms
        "moderate", "mild", "average", "typical", "standard", "conventional", 
        "unremarkable", "expected", "predictable", "uniform"
    ],
    
    "Positive": [
        # Economic Growth
        "growth", "expansion", "economic boom", "recovery", "prosperity", "economic surge", 
        "economic renaissance", "economic breakthrough", "economic acceleration",
        
        # Financial Performance
        "profit", "revenue growth", "record earnings", "financial success", 
        "exceeded expectations", "strong performance", "financial breakthrough", 
        "outstanding results", "exceptional performance",
        
        # Market Indicators
        "bull market", "market rally", "stock surge", "market boom", "market expansion", 
        "market optimization", "market leadership", "market dominance",
        
        # Job Market
        "job creation", "hiring spree", "workforce expansion", "employment growth", 
        "talent acquisition", "skill development", "career opportunities",
        
        # Monetary Positive
        "currency appreciation", "economic stimulus", "quantitative easing", 
        "investment inflow", "capital injection", "financial incentive", 
        "economic empowerment",
        
        # Business Success
        "innovation", "breakthrough", "market disruption", "technological advancement", 
        "business expansion", "new market entry", "strategic acquisition", 
        "successful merger",
        
        # Investment Positive
        "investment growth", "portfolio expansion", "asset appreciation", 
        "positive return", "high-yield investment", "strategic investment", 
        "investment opportunity",
        
        # Positive Financial Terms
        "upgrade", "positive outlook", "credit rating upgrade", "sector leadership", 
        "regulatory support", "financial innovation", "economic transformation",
        
        # Positive Performance Words
        "boost", "surge", "climb", "rise", "grow", "expand", "increase", "improve", 
        "strengthen", "accelerate", "optimize", "breakthrough", "revolutionize",
        
        # Specific Financial Achievements
        "trade surplus", "budget surplus", "positive trade balance", "capital gain", 
        "positive cash flow", "debt reduction", "financial restructuring",
        
        # Global Economic Positive
        "global economic recovery", "emerging market opportunity", 
        "international trade expansion", "economic collaboration", 
        "cross-border innovation",
        
        
        "tech sector innovation", "retail renaissance", "energy sector transformation", 
        "real estate market recovery", "banking innovation", "automotive industry revival",
        
        
        "potential", "opportunity", "promising", "future-ready", "strategic", 
        "competitive advantage", "market potential", "innovative approach"
    ]
        }

    def _advanced_rule_based_sentiment(self, sentence):
        """
        Enhanced rule-based sentiment analysis with regex and context-aware matching
        """
        sentence_lower = sentence.lower()
        
        
        sentiment_scores = {
            "Negative": 0,
            "Neutral": 0,
            "Positive": 0
        }
        
        
        for sentiment, keywords in self.rule_based_keywords.items():
            for keyword in keywords:
                matches = re.findall(r'\b' + keyword + r'\b', sentence_lower)
                sentiment_scores[sentiment] += len(matches)
        
        
        max_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        max_score = sentiment_scores[max_sentiment]
        
        # Only return if there's a significant match
        return max_sentiment if max_score > 0 else None

    def sentiment_analysis(self, sentences):
        results = []
        for sentence in sentences:
            # First, try advanced rule-based classification
            rule_sentiment = self._advanced_rule_based_sentiment(sentence)
            
            if rule_sentiment:
                results.append({
                    "sentence": sentence, 
                    "sentiment": rule_sentiment
                })
            else:
                # Fallback to machine learning model
                inputs = self.tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=1).item()
                
                # Map numerical labels back to sentiments
                label_map_reverse = {0: "Negative", 1: "Neutral", 2: "Positive"}
                model_sentiment = label_map_reverse[prediction]
                
                results.append({
                    "sentence": sentence, 
                    "sentiment": model_sentiment
                })
        
        return results


analyzer = EnhancedSentimentAnalyzer()
test_sentences = [
    "US President Trump is shot dead"
]

# Run sentiment analysis
analysis_results = analyzer.sentiment_analysis(test_sentences)

# Display results
for result in analysis_results:
    print(f"Sentence: {result['sentence']}")
    print(f"Predicted Sentiment: {result['sentiment']}")
    print()