import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
import requests
from datetime import datetime, timedelta

class StockPricePredictionModel:
    def __init__(self, stocks_list):
        """
        Initialize the stock prediction model
        
        :param stocks_list: List of stock tickers
        """
        self.stocks_list = [stock.replace('.NS', '') for stock in stocks_list]
        self.stock_data = {}
        self.news_impact = {}
    
    def fetch_stock_data(self, days=365):
        """
        Fetch historical stock data for listed stocks
        
        :param days: Number of days of historical data to fetch
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for stock in self.stocks_list:
            try:
                # Add .NS suffix for NSE stocks
                full_ticker = f"{stock}.NS"
                stock_info = yf.Ticker(full_ticker)
                
                # Fetch historical data
                history = stock_info.history(start=start_date, end=end_date)
                
                if not history.empty:
                    self.stock_data[stock] = history
                    print(f"Successfully fetched data for {stock}")
                else:
                    print(f"No data available for {stock}")
            except Exception as e:
                print(f"Error fetching data for {stock}: {e}")
    
    def _get_stock_sector(self, stock):
        """
        Determine stock sector based on predefined mapping
        
        :param stock: Stock ticker
        :return: Sector of the stock
        """
        sector_mapping = {
            'defense': ['HAL', 'BDL'],
            'technology': ['INFY', 'TCS', 'WIPRO'],
            'banking': ['ICICI', 'HDFC', 'SBI'],
            'oil': ['RELIANCE'],
            'pharmaceuticals': ['SUNPHARMA', 'DRREDDY'],
            'automotive': ['MARUTI', 'HEROMOTOCO']
        }
        
        for sector, stocks in sector_mapping.items():
            if stock in stocks:
                return sector
        
        return 'general'
    
    def simulate_news_sentiment(self):
        """
        Simulate news sentiment for stocks
        Replaces actual news API with a basic simulation
        """
        news_impacts = {}
        
        for stock in self.stocks_list:
            sector = self._get_stock_sector(stock)
            
            # Simulate sentiment based on sector and recent stock performance
            if stock in self.stock_data:
                stock_data = self.stock_data[stock]
                price_change = (stock_data['Close'][-1] - stock_data['Close'][0]) / stock_data['Close'][0] * 100
                
                # Simulate sentiment with some randomness and price change influence
                base_sentiment = {
                    'defense': 0.7,
                    'technology': 0.3,
                    'banking': 0.1,
                    'oil': 0.5,
                    'pharmaceuticals': 0.2,
                    'automotive': 0.4,
                    'general': 0
                }.get(sector, 0)
                
                # Adjust sentiment based on price change
                adjusted_sentiment = base_sentiment + (price_change / 100)
                
                news_impacts[stock] = {
                    'avg_sentiment': adjusted_sentiment,
                    'sector': sector,
                    'price_change': price_change
                }
            else:
                news_impacts[stock] = {
                    'avg_sentiment': 0,
                    'sector': sector,
                    'price_change': 0
                }
        
        self.news_impact = news_impacts
    
    def predict_stock_recommendations(self):
        """
        Generate stock recommendations based on simulated news sentiment
        
        :return: Recommendation dictionary
        """
        recommendations = {}
        
        for stock, news_data in self.news_impact.items():
            sentiment = news_data['avg_sentiment']
            price_change = news_data['price_change']
            
            # Enhanced recommendation logic
            if sentiment > 0.5 and price_change > 0:
                recommendation = 'STRONG BUY'
                confidence = min((sentiment + abs(price_change/10)) * 100, 100)
            elif sentiment > 0.2 and price_change > 0:
                recommendation = 'BUY'
                confidence = min((sentiment + abs(price_change/20)) * 100, 85)
            elif sentiment < -0.5 and price_change < 0:
                recommendation = 'STRONG SELL'
                confidence = min((abs(sentiment) + abs(price_change/10)) * 100, 100)
            elif sentiment < -0.2 and price_change < 0:
                recommendation = 'SELL'
                confidence = min((abs(sentiment) + abs(price_change/20)) * 100, 85)
            else:
                recommendation = 'HOLD'
                confidence = abs(sentiment) * 50
            
            recommendations[stock] = {
                'recommendation': recommendation,
                'confidence': max(0, min(confidence, 100)),
                'sector': news_data['sector'],
                'price_change': f"{price_change:.2f}%"
            }
        
        return recommendations

def main():
    # Updated list of top Indian stocks
    top_stocks = [
        'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI', 
        'SBIN', 'BAJFINANCE', 'KOTAKBANK', 'HDFCBANK', 
        'MARUTI', 'HEROMOTOCO', 'WIPRO', 'SUNPHARMA', 'DRREDDY'
    ]
    
    # Create and run the model
    model = StockPricePredictionModel(top_stocks)
    
    # Fetch stock data
    model.fetch_stock_data()
    
    # Simulate news sentiment
    model.simulate_news_sentiment()
    
    # Generate recommendations
    recommendations = model.predict_stock_recommendations()
    
    # Print recommendations
    print("\n--- Stock Recommendations ---")
    for stock, rec in recommendations.items():
        print(f"{stock}: {rec['recommendation']} - {rec['confidence']:.2f}% Confidence")
        print(f"  Sector: {rec['sector']}")
        print(f"  Price Change: {rec['price_change']}\n")

if __name__ == "__main__":
    main()