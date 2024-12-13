import random
import pandas as pd

# Expanded list of markets and events
markets = ["US", "China", "Europe", "Japan", "India"]
events = {
    "US": [
        "Federal Reserve announces interest rate hike",
        "Tech giants report record profits",
        "Trade war with China escalates",
        "US dollar strengthens against major currencies",
        "Corporate tax reforms boost FDI in emerging markets",
    ],
    "China": [
        "Manufacturing PMI drops to a 5-year low",
        "Export restrictions on rare earth metals announced",
        "Trade deal with Europe signed, bypassing US tariffs",
        "Economic slowdown concerns ripple through Asia",
        "Tech breakthroughs in AI challenge Indian IT dominance",
    ],
    "Europe": [
        "ECB announces quantitative easing measures",
        "Energy crisis due to Middle East tensions",
        "Political instability in key economies affects global markets",
        "Indian exports benefit from reduced European tariffs",
        "Brexit uncertainty impacts global trade",
    ],
    "Japan": [
        "Yen weakens, boosting Japanese exports",
        "India signs technology transfer agreement with Japan",
        "Japanese investors focus on Indian infrastructure projects",
        "Bank of Japan announces continued monetary easing",
        "Japanese automakers report increased demand from India",
    ],
    "India": [
        "GDP growth exceeds expectations",
        "RBI cuts repo rate to spur growth",
        "Trade surplus with US increases due to IT exports",
        "Indian markets rally on global capital inflows",
        "Rupee strengthens against major currencies",
    ],
}

# Positive, Negative, and Neutral impacts per market interaction
impact_map = {
    "Positive": [
        "boosts Indian exports significantly",
        "leads to increased FDI inflows into India",
        "creates opportunities for Indian startups",
        "strengthens trade ties between India and key markets",
        "improves India's economic outlook globally",
    ],
    "Negative": [
        "increases inflation risks for the Indian economy",
        "disrupts supply chains critical for Indian industries",
        "causes capital outflows from Indian markets",
        "leads to currency depreciation against the US dollar",
        "reduces demand for Indian exports",
    ],
    "Neutral": [
        "has mixed effects on the Indian economy",
        "leads to cautious optimism among Indian businesses",
        "requires India to reassess trade strategies",
        "creates temporary volatility in Indian markets",
        "results in no significant impact on Indian growth",
    ],
}

# Generate the dataset
def generate_global_market_dataset(num_samples=10000):
    data = []
    for _ in range(num_samples):
        market = random.choice(markets)
        event = random.choice(events[market])
        sentiment = random.choice(["Positive", "Negative", "Neutral"])
        impact = random.choice(impact_map[sentiment])
        
        # Construct headline
        headline = f"{market}: {event} {impact}"
        
        # Append to dataset
        data.append({"headline": headline, "sentiment": sentiment})
    
    return pd.DataFrame(data)

# Generate large dataset with global interactions
global_market_dataset = generate_global_market_dataset()
global_market_dataset.to_csv("global_market_interactions.csv", index=False)
print(global_market_dataset.head())