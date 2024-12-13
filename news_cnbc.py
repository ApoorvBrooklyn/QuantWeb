import pandas as pd
from datetime import datetime, timedelta
import re
import ycnbc

news = ycnbc.News()

def clean_headline(headline):
    """Cleans a single headline by fixing encoding issues."""
    try:
        return headline.encode('latin1').decode('utf-8').strip()
    except (UnicodeEncodeError, UnicodeDecodeError):
        return headline.strip()

def parse_time(timestamp):
    """Parses the time into a standardized datetime format."""
    try:
        if 'ago' in timestamp.lower():
            match = re.search(r"(\d+)\s*hours\s*ago", timestamp)
            if match:
                hours_ago = int(match.group(1))
                return datetime.now() - timedelta(hours=hours_ago)
            return datetime.now()
        else:
            # Adjust for any variation in date format (e.g., "Thu, Dec 12th 2024")
            return datetime.strptime(timestamp, '%a, %b %dth %Y')
    except ValueError:
        return None

def extract_tickers(headline):
    """Extracts potential stock tickers from headlines using a regex pattern."""
    ticker_pattern = r'\b[A-Z]{2,5}\b'
    return re.findall(ticker_pattern, headline)

def preprocess_news_data(raw_data):
    """Preprocesses raw news data to clean and structure it."""
    headlines = raw_data.get('headline', [])
    times = raw_data.get('time', [])
    links = raw_data.get('link', [])

    # Ensure lengths match
    if not (len(headlines) == len(times) == len(links)):
        print("Warning: Mismatch in data lengths. Some data might be missing.")

    processed_data = []
    for headline, timestamp, link in zip(headlines, times, links):
        cleaned_headline = clean_headline(headline)
        parsed_time = parse_time(timestamp)
        tickers = extract_tickers(cleaned_headline)

        processed_data.append({
            'headline': cleaned_headline,
            'timestamp': parsed_time,
            'tickers': tickers,
            'link': link
        })

    return pd.DataFrame(processed_data)

# Fetch data
Data = news.economy()

# Process the data
processed_news = preprocess_news_data(Data)

# Display results
print(processed_news)