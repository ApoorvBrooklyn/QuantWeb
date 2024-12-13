import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('.')  # Ensure current directory is in path
from Nlp import EnhancedSentimentAnalyzer
from news_cnbc import processed_news

from Nlp import headlines

headline = headlines.iloc[0]

class AdvancedStockAnalysis:
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize stock analysis with comprehensive technical indicators
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date for data collection
            end_date (str): End date for data collection
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        # Fetch stock data
        self.stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Ensure data is available
        if self.stock_data.empty:
            raise ValueError(f"No data available for {ticker} between {start_date} and {end_date}")
        
        # Sentiment Analyzer
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
    
    def calculate_technical_indicators(self):
        """
        Calculate comprehensive technical indicators
        
        Returns:
            pandas.DataFrame: Stock data with technical indicators
        """
        df = self.stock_data.copy()
        
        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Ensure 20-day moving average exists
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Bollinger Bands
        df['STD_20'] = df['Close'].rolling(window=20).std()
        df['BOLL_UPPER'] = df['SMA_20'] + (df['STD_20'] * 2)
        df['BOLL_LOWER'] = df['SMA_20'] - (df['STD_20'] * 2)
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['SIGNAL_LINE'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        df['ATR_14'] = true_range.rolling(window=14).mean()
        
        return df

    def analyze_sentiment(self, headline):
        """
        Analyze sentiment of news articles
        
        Args:
            news_articles (list): List of news articles about the stock
        
        Returns:
            dict: Sentiment analysis results
        """
        # Use model's default settings 
        return self.sentiment_analyzer.sentiment_analysis(headline)
    
    
    
    def identify_trend_and_resistance(self, df):
        """
        Identify stock trend and resistance levels
        
        Args:
            df (pandas.DataFrame): Stock data with technical indicators
        
        Returns:
            dict: Trend and resistance analysis
        """
        # Trend Identification
        try:
            # Ensure we have at least 2 data points
            if len(df) < 2:
                return {
                    'trend': 'Insufficient Data',
                    'resistance_levels': []
                }

            # Use explicit numeric comparison
            first_close = df['Close'].iloc[0]
            lastsingle_headline = headline.iloc[0]
            resistance_levels = []
            lookback_windows = [50, 100, 200]
            
            for window in lookback_windows:
                # Adjust window size if not enough data
                adjusted_window = min(window, len(df))
                
                try:
                    # Calculate resistance levels
                    window_data = df.tail(adjusted_window)
                    high = window_data['High'].max()
                    
                    resistance_levels.append({
                        'lookback': adjusted_window,
                        'resistance_level': float(high),
                        'current_price': float(df['Close'].iloc[-1]),
                        'distance_from_resistance': float(high - df['Close'].iloc[-1])
                    })
                except Exception as window_error:
                    print(f"Error calculating resistance for window {window}: {window_error}")
                    resistance_levels.append({
                        'lookback': adjusted_window,
                        'error': str(window_error)
                    })
            
            return {
                'trend': trend,
                'resistance_levels': resistance_levels
            }

        except Exception as e:
            # Comprehensive error handling
            print(f"Error in trend identification: {e}")
            return {
                'trend': 'Error',
                'error': str(e),
                'resistance_levels': []
            }

    def analyze_sentiment(self, news_articles):
        """
        Analyze sentiment of news articles with enhanced error handling
        
        Args:
            news_articles (list): List of news articles about the stock
        
        Returns:
            dict: Sentiment analysis results
        """
        try:
            # Ensure news_articles is a list
            if not isinstance(news_articles, list):
                news_articles = [str(news_articles)]
            
            # Ensure non-empty articles
            news_articles = [str(article).strip() for article in news_articles if article]
            
            # If no articles after filtering
            if not news_articles:
                return {
                    'overall_sentiment': 'Neutral',
                    'sentiment_score': 0,
                    'error': 'No valid articles provided'
                }
            
            # Use model's default settings with error handling
            sentiment_results = self.sentiment_analyzer.sentiment_analysis(headline)
            
            # Validate and standardize results
            if not isinstance(sentiment_results, dict):
                return {
                    'overall_sentiment': 'Neutral',
                    'sentiment_score': 0,
                    'error': 'Invalid sentiment analysis result'
                }
            
            return {
                'overall_sentiment': sentiment_results.get('overall_sentiment', 'Neutral'),
                'sentiment_score': sentiment_results.get('sentiment_score', 0)
            }
        
        except Exception as e:
            return {
                'overall_sentiment': 'Neutral',
                'sentiment_score': 0,
                'error': str(e)
            }

    
    def generate_investment_insight(self, technical_df, sentiment_results):
        """
        Generate comprehensive investment insights based on technical analysis and sentiment
        
        Args:
            technical_df (pandas.DataFrame): Technical indicators dataframe
            sentiment_results (dict): Sentiment analysis results
        
        Returns:
            dict: Comprehensive investment insights
        """
        # Get the most recent data point
        latest_data = technical_df.iloc[-1]
        
        # Trend and Resistance Analysis
        trend_analysis = self.identify_trend_and_resistance(technical_df)
        
        # Investment Recommendation Logic
        # Combine technical indicators and sentiment for insights
        investment_insights = {
            'ticker': self.ticker,
            'current_price': latest_data['Close'].item(),  # Ensure it's a scalar
            'trend': trend_analysis['trend'],
            'technical_indicators': {
                'RSI': latest_data['RSI'].item(),  # Ensure it's a scalar
                'MACD': latest_data['MACD'].item(),  # Ensure it's a scalar
                'signal_line': latest_data['SIGNAL_LINE'].item(),  # Ensure it's a scalar
                'SMA_10': latest_data['SMA_10'].item(),  # Ensure it's a scalar
                'SMA_50': latest_data['SMA_50'].item()   # Ensure it's a scalar
            },
            'resistance_levels': [
                {
                    'lookback': level['lookback'],
                    'resistance_level': float(level['resistance_level']),
                    'current_price': float(level['current_price']),
                    'distance_from_resistance': float(level['distance_from_resistance'])
                } for level in trend_analysis['resistance_levels']
            ],
            'sentiment': {
                'overall_sentiment': sentiment_results.get('overall_sentiment', 'Neutral'),
                'sentiment_score': sentiment_results.get('sentiment_score', 0)
            },
            'recommendation': self._determine_recommendation(
                technical_df, 
                trend_analysis, 
                sentiment_results
            )
        }
        
        return investment_insights

    
    def _determine_recommendation(self, technical_df, trend_analysis, sentiment_results):

        """

        Determine investment recommendation based on technical and sentiment analysis.

        """

        latest_data = technical_df.iloc[-1]

        

        # Extract scalar values for calculations

        rsi = latest_data['RSI'].item()  # Ensure it's a scalar

        macd = latest_data['MACD'].item()  # Ensure it's a scalar

        signal_line = latest_data['SIGNAL_LINE'].item()  # Ensure it's a scalar

        sentiment_score = sentiment_results.get('sentiment_score', 0)

        

        # Recommendation Logic

        if trend_analysis['trend'] == 'Uptrend':

            if (rsi < 30) and (macd > signal_line) and (sentiment_score > 0):

                return 'Strong Buy'

            elif (rsi < 40) and (macd > signal_line):

                return 'Buy'

            else:

                return 'Hold'

        else:  # Downtrend

            if (rsi > 70) and (macd < signal_line) and (sentiment_score < 0):

                return 'Strong Sell'

            elif (rsi > 60) and (macd < signal_line):

                return 'Sell'

            else:

                return 'Hold'

    def visualize_analysis(self, technical_df):
        """
        Visualize stock analysis with key technical indicators
        
        Args:
            technical_df (pandas.DataFrame): Technical indicators dataframe
        """
        plt.figure(figsize=headline(12, 8))
        
        # Price and Moving Averages
        plt.subplot(2, 1, 1)
        plt.plot(technical_df.index, technical_df['Close'], label='Close Price')
        plt.plot(technical_df.index, technical_df['SMA_10'], label='10-day SMA')
        plt.plot(technical_df.index, technical_df['SMA_50'], label='50-day SMA')
        plt.title(f'{self.ticker} Stock Price and Moving Averages')
        plt.legend()
        
        # Technical Indicators
        plt.subplot(2, 1, 2)
        plt.plot(technical_df.index, technical_df['RSI'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.title('Relative Strength Index (RSI)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # Example Usage
    ticker = 'LICI.NS'  # Example stock
    start_date = '2023-01-01'
    end_date = '2024-10-01'
    
    
    
    # Create stock analysis instance
    stock_analysis = AdvancedStockAnalysis(ticker, start_date, end_date)
    
    # Calculate technical indicators
    technical_df = stock_analysis.calculate_technical_indicators()
    
    # Analyze sentiment
    sentiment_results = stock_analysis.analyze_sentiment(headline)
    
    # Generate investment insights
    investment_insights = stock_analysis.generate_investment_insight(
        technical_df, 
        sentiment_results
    )
    
    # Print insights
    print(f"Investment Insights for {ticker}:")
    print(json.dumps(investment_insights, indent=2))
    
    # Visualize analysis
    stock_analysis.visualize_analysis(technical_df)

if __name__ == "__main__":
    main()