import yfinance as yf
import numpy as np
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta
import ycnbc

class StockPredictionModel:
    def __init__(self, stocks_list):
        """
        Initialize the stock prediction model
        
        :param stocks_list: List of stock tickers
        """
        self.stocks_list = stocks_list
        self.stock_data = {}
        self.news_data = {}
        self.sentiment_analysis = {}
    
    def process_news(self, news_data):
        """
        Process news data from YCNBC
        
        :param news_data: Dictionary containing headlines and links
        """
        # Store raw news data
        self.news_data = news_data
        
        # Analyze sentiment of news
        self._analyze_news_sentiment()
    
    def _analyze_news_sentiment(self):
        """
        Analyze sentiment of news headlines
        """
        # Initialize sector sentiments
        sector_sentiments = {
            'healthcare': [],
            'technology': [],
            'finance': [],
            'politics': [],
            'business': [],
            'general': []
        }
        
        # Analyze each headline
        for headline in self.news_data.get('headline', []):
            # Perform sentiment analysis
            blob = TextBlob(headline)
            sentiment = blob.sentiment.polarity
            
            # Categorize sentiment based on keywords
            self._categorize_news_sentiment(headline, sentiment, sector_sentiments)
        
        # Calculate average sentiments
        self.sentiment_analysis = {
            sector: np.mean(sentiments) if sentiments else 0 
            for sector, sentiments in sector_sentiments.items()
        }
    
    def _categorize_news_sentiment(self, text, sentiment, sector_sentiments):
        """
        Categorize news sentiment based on keywords
        
        :param text: News headline
        :param sentiment: Sentiment score
        :param sector_sentiments: Dictionary to store sector sentiments
        """
        # Lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Keyword mappings for sectors
        sector_keywords = {
            'healthcare': ['health', 'hospital', 'medical', 'doctor', 'united', 'treatment'],
            'technology': ['tech', 'broadcom', 'elon', 'musk', 'twitter', 'stock', 'market cap'],
            'finance': ['stock', 'market', 'billion', 'ceo', 'company', 'worth'],
            'politics': ['nancy', 'pelosi', 'sec', 'settlement'],
            'business': ['kfc', 'franchise', 'company']
        }
        
        # Check and assign to sectors
        for sector, keywords in sector_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                sector_sentiments[sector].append(sentiment)
                return
        
        # If no specific sector found, add to general
        sector_sentiments['general'].append(sentiment)
    
    def fetch_stock_data(self, days=30):
        """
        Fetch historical stock data for stocks mentioned in news
        
        :param days: Number of days of historical data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Extract unique stocks from headlines
        mentioned_stocks = self._extract_stocks_from_news()
        
        for stock in mentioned_stocks:
            try:
                # Fetch stock data
                stock_info = yf.Ticker(stock)
                history = stock_info.history(start=start_date, end=end_date)
                
                if not history.empty:
                    self.stock_data[stock] = history
            except Exception as e:
                print(f"Error fetching data for {stock}: {e}")
    
    def _extract_stocks_from_news(self):
        """
        Extract stock tickers mentioned in news headlines
        
        :return: List of stock tickers
        """
        # Predefined mapping of company names to stock tickers
        stock_mapping = {
            'unitedhealth': 'UNH',
            'broadcom': 'AVGO',
            'elon musk': 'TSLA',
            'twitter': 'TWTR',
            'kfc': 'QSR'
        }
        
        mentioned_stocks = []
        for headline in self.news_data.get('headline', []):
            headline_lower = headline.lower()
            
            for company, ticker in stock_mapping.items():
                if company in headline_lower:
                    mentioned_stocks.append(ticker)
        
        return mentioned_stocks
    
    def generate_recommendations(self):
        """
        Generate stock recommendations based on news sentiment and stock performance
        
        :return: Comprehensive recommendation report
        """
        recommendations = {}
        
        for stock in self.stock_data.keys():
            # Get stock history
            stock_history = self.stock_data[stock]
            
            # Calculate price change
            price_change = (stock_history['Close'][-1] - stock_history['Close'][0]) / stock_history['Close'][0] * 100
            
            # Determine stock's sector
            sector = self._determine_stock_sector(stock)
            
            # Get sector sentiment
            sector_sentiment = self.sentiment_analysis.get(sector, 0)
            
            # Recommendation calculation
            buy_percent, hold_percent, sell_percent = self._calculate_recommendation(sector_sentiment, price_change)
            
            recommendations[stock] = {
                'sector': sector,
                'price_change': f"{price_change:.2f}%",
                'news_sentiment': f"{sector_sentiment:.2f}",
                'recommendation': {
                    'buy': buy_percent,
                    'hold': hold_percent,
                    'sell': sell_percent
                }
            }
        
        return recommendations
    
    def _calculate_recommendation(self, sentiment, price_change):
        """
        Calculate recommendation percentages based on sentiment and price change
        
        :param sentiment: Sector sentiment score
        :param price_change: Stock price change percentage
        :return: Tuple of (buy%, hold%, sell%)
        """
        # Base recommendation logic
        if sentiment > 0.5 and price_change > 0:
            return 70, 25, 5
        elif sentiment > 0 and price_change > 0:
            return 55, 35, 10
        elif sentiment < -0.5 and price_change < 0:
            return 10, 20, 70
        elif sentiment < 0 and price_change < 0:
            return 20, 30, 50
        else:
            return 40, 40, 20
    
    def _determine_stock_sector(self, stock):
        """
        Determine stock sector based on predefined mapping
        
        :param stock: Stock ticker
        :return: Sector of the stock
        """
        sector_mapping = {
            'healthcare': ['UNH'],
            'technology': ['AVGO', 'TSLA'],
            'finance': ['TWTR'],
            'business': ['QSR']
        }
        
        for sector, stocks in sector_mapping.items():
            if stock in stocks:
                return sector
        
        return 'general'
    
    def run_analysis(self, news_data):
        """
        Run complete stock prediction analysis
        
        :param news_data: News data from YCNBC
        :return: Stock recommendations
        """
        # Process news
        self.process_news(news_data)
        
        # Fetch stock data
        self.fetch_stock_data()
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        return recommendations

def main():
 

    news = ycnbc.News()
    #getting latest news
    latest = news.latest()
    news_data = {
        'headline':latest['headline']
    }
    
    # Create and run the model
    model = StockPredictionModel([])
    
    # Run analysis
    recommendations = model.run_analysis(news_data)
    
    # Print recommendations
    print("\n--- Stock Recommendations ---")
    for stock, rec in recommendations.items():
        print(f"\nStock: {stock}")
        print(f"Sector: {rec['sector']}")
        print(f"Price Change: {rec['price_change']}")
        print(f"News Sentiment: {rec['news_sentiment']}")
        print("Recommendation Breakdown:")
        print(f"  Buy: {rec['recommendation']['buy']}%")
        print(f"  Hold: {rec['recommendation']['hold']}%")
        print(f"  Sell: {rec['recommendation']['sell']}%")

if __name__ == "__main__":
    main()