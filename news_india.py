import ycnbc

# Initialize the news API
news = ycnbc.News()

# Fetch finance-related news
finance_news = news.finance()

# Keywords to filter Indian stock market-related news
keywords = ["India", "NSE", "BSE", "Sensex", "Nifty"]

# Filter news articles
indian_stock_market_news = [
    article for article in finance_news
    if any(keyword in (article.get("title", "") + article.get("description", "")) for keyword in keywords)
]

# Display the filtered news
for article in indian_stock_market_news:
    print(f"Title: {article.get('title')}")
    print(f"Description: {article.get('description')}")
    print(f"Link: {article.get('url')}")
    print("\n")
