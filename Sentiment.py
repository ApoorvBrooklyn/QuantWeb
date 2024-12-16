from transformers import pipeline

# Create a sentiment analysis pipeline
pipe = pipeline(
    "sentiment-analysis",
    model="StephanAkkerman/FinTwitBERT-sentiment",
)

# Get the predicted sentiment
print(pipe("US president shot dead"))
