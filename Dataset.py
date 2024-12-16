import pandas as pd

# Extend the synthetic dataset with new examples
extended_data = [
    {"News_Headline": "US president killed by terrorists", "Sentiment": "Negative"},
    {"News_Headline": "Tensions escalate between US and China over trade", "Sentiment": "Negative"},
    {"News_Headline": "Breakthrough peace deal signed in Middle East", "Sentiment": "Positive"},
    {"News_Headline": "Global oil prices surge as major supplier cuts output", "Sentiment": "Negative"},
    {"News_Headline": "Tech giant reports record-breaking quarterly earnings", "Sentiment": "Positive"},
    {"News_Headline": "Earthquake disrupts manufacturing hub in Japan", "Sentiment": "Negative"},
    {"News_Headline": "Federal Reserve announces unexpected interest rate hike", "Sentiment": "Negative"},
    {"News_Headline": "Major airline declares bankruptcy amid rising fuel costs", "Sentiment": "Negative"},
    {"News_Headline": "Government launches massive infrastructure stimulus program", "Sentiment": "Positive"},
    {"News_Headline": "Global markets rally as inflation cools down", "Sentiment": "Positive"},
    {"News_Headline": "Cyberattack hits major financial institution", "Sentiment": "Negative"},
    {"News_Headline": "Leading pharmaceutical company announces cure for major disease", "Sentiment": "Positive"},
    {"News_Headline": "Unexpected resignation of top tech CEO shocks investors", "Sentiment": "Negative"},
    {"News_Headline": "New environmental regulations boost green energy stocks", "Sentiment": "Positive"},
    {"News_Headline": "Mass protests erupt across Europe due to rising living costs", "Sentiment": "Negative"}
]

# Load existing dataset
original_data = pd.read_csv("synthetic_financial_sentiment_dataset.csv")

# Append the new examples
new_data = pd.DataFrame(extended_data)
combined_data = pd.concat([original_data, new_data], ignore_index=True)

# Save the extended dataset
combined_data.to_csv("extended_financial_sentiment_dataset.csv", index=False)
print("Extended dataset saved as 'extended_financial_sentiment_dataset.csv'")