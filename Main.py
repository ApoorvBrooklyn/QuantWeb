import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('.')  # Ensure current directory is in path

class EnhancedSentimentAnalyzer:
    def sentiment_analysis(self, text):
        """
        Simple sentiment analysis with more nuanced scoring
    
        Args:
            text (str): Text to analyze
        
        Returns:
            dict: Sentiment analysis results
        """
        # Basic keyword-based sentiment scoring
        positive_keywords = ['growth', 'opportunity', 'strong', 'positive', 'bullish']
        negative_keywords = ['decline', 'risk', 'weak', 'negative', 'bearish']
        
        text_lower = text.lower()
        
        positive_count = sum(keyword in text_lower for keyword in positive_keywords)
        negative_count = sum(keyword in text_lower for keyword in negative_keywords)
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count + 1)
        
        # Determine overall sentiment
        if sentiment_score > 0.3:
            overall_sentiment = 'Positive'
        elif sentiment_score < -0.3:
            overall_sentiment = 'Negative'
        else:
            overall_sentiment = 'Neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': sentiment_score
        }

class AdvancedStockAnalysis:
    def __init__(self, ticker, start_date, end_date, headline="Default headline"):
        """
        Initialize stock analysis with comprehensive technical indicators
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date for data collection
            end_date (str): End date for data collection
            headline (str, optional): News headline for sentiment analysis
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.headline = headline
        
        # Fetch stock data with extended history
        extended_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=100)
        self.stock_data = yf.download(ticker, start=extended_start_date, end=end_date)
        
        # Ensure data is available
        if self.stock_data.empty:
            raise ValueError(f"No data available for {ticker} between {start_date} and {end_date}")
        
        # Sentiment Analyzer
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
    
    def calculate_technical_indicators(self):
        """
        Calculate comprehensive technical indicators with robust fallback
        
        Returns:
            pandas.DataFrame: Stock data with technical indicators
        """
        df = self.stock_data.copy()
        
        # Robust moving averages with fallback
        def safe_rolling(series, window):
            return series.rolling(window=min(window, len(series))).mean()
        
        # Moving Averages
        df['SMA_10'] = safe_rolling(df['Close'], 10)
        df['SMA_50'] = safe_rolling(df['Close'], 50)
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Relative Strength Index (RSI) with more robust calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / (loss + 1e-10)  # Prevent divide by zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD with safety checks
        try:
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['SIGNAL_LINE'] = df['MACD'].ewm(span=9, adjust=False).mean()
        except Exception as e:
            print(f"MACD calculation error: {e}")
            df['MACD'] = 0
            df['SIGNAL_LINE'] = 0
        
        return df.dropna()  # Remove rows with NaN to ensure clean data

    def _determine_recommendation(self, technical_df, sentiment_results):
        """
        Enhanced recommendation determination with more nuanced logic
        
        Args:
            technical_df (pandas.DataFrame): Technical indicators dataframe
            sentiment_results (dict): Sentiment analysis results
        
        Returns:
            str: Investment recommendation
        """
        latest_data = technical_df.iloc[-1]

        # Safely extract values with .item()
        rsi = latest_data['RSI'].item()
        macd = latest_data['MACD'].item()
        signal_line = latest_data.get('SIGNAL_LINE', 0)
        if isinstance(signal_line, pd.Series):
            signal_line = signal_line.item()
        sentiment_score = sentiment_results.get('sentiment_score', 0)

        # More nuanced recommendation logic
        if rsi < 30 and macd > signal_line:
            return 'Strong Buy' if sentiment_score > 0.5 else 'Buy'
        elif rsi > 70 and macd < signal_line:
            return 'Strong Sell' if sentiment_score < -0.5 else 'Sell'
        elif rsi < 40 and macd > signal_line:
            return 'Potential Buy'
        elif rsi > 60 and macd < signal_line:
            return 'Potential Sell'
        else:
            # Consider sentiment for neutral scenarios
            if sentiment_score > 0.3:
                return 'Cautious Buy'
            elif sentiment_score < -0.3:
                return 'Cautious Sell'
            else:
                return 'Hold'

    def generate_investment_insight(self):
        """
        Generate comprehensive investment insights
        
        Returns:
            dict: Comprehensive investment insights
        """
        # Calculate technical indicators
        technical_df = self.calculate_technical_indicators()
        
        # Analyze sentiment
        sentiment_results = self.analyze_sentiment()
        
        # Get the most recent data point
        latest_data = technical_df.iloc[-1]
        
        # Investment Recommendation Logic
        investment_insights = {
            'ticker': self.ticker,
            'current_price': latest_data['Close'].item(),  # Use .item() to convert to scalar
            'technical_indicators': {
                'RSI': latest_data['RSI'].item(),
                'MACD': latest_data['MACD'].item(),
                'signal_line': latest_data.get('SIGNAL_LINE', 0).item() if isinstance(latest_data.get('SIGNAL_LINE', 0), pd.Series) else latest_data.get('SIGNAL_LINE', 0),
            },
            'sentiment': sentiment_results,
            'recommendation': self._determine_recommendation(
                technical_df, 
                sentiment_results
            )
        }
        
        return investment_insights

    def analyze_sentiment(self, news_article=None):
        """
        Analyze sentiment of news articles
        
        Args:
            news_article (str, optional): News article about the stock
        
        Returns:
            dict: Sentiment analysis results
        """
        try:
            # Use provided news_article or default to self.headline
            news_article = news_article or self.headline
            
            # Validate input
            if not news_article or not isinstance(news_article, str):
                return {
                    'overall_sentiment': 'Neutral',
                    'sentiment_score': 0
                }
            
            # Use enhanced sentiment analysis
            return self.sentiment_analyzer.sentiment_analysis(news_article)
        
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {
                'overall_sentiment': 'Neutral',
                'sentiment_score': 0
            }

def main():
    # Example Usage
    ticker = 'ZOMATO.NS'
    start_date = '2023-01-01'  # Extended historical data to ensure enough points
    end_date = '2024-11-28'
    
    # More descriptive and varied headlines
    headlines = [
        "Zomato shows strong growth in food delivery market",
        "Challenges persist in restaurant tech sector",
        "Innovative strategies boost Zomato's market position"
    ]
    
    for headline in headlines:
        # Create stock analysis instance
        stock_analysis = AdvancedStockAnalysis(
            ticker, 
            start_date, 
            end_date, 
            headline=headline
        )
        
        try:
            # Generate investment insights
            investment_insights = stock_analysis.generate_investment_insight()
            
            # Print insights
            print(f"\nInvestment Insights for {ticker} with headline: {headline}")
            print(json.dumps(investment_insights, indent=2))
        
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    main()