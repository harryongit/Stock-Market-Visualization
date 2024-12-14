import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

class StockMarketAnalytics:
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize the stock market analytics class
        
        :param ticker: Stock ticker symbol
        :param start_date: Start date for data retrieval
        :param end_date: End date for data retrieval
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.predictions = None
    
    def fetch_stock_data(self):
        """
        Retrieve stock data using yfinance
        """
        self.data = yf.download(self.ticker, 
                                start=self.start_date, 
                                end=self.end_date)
        return self.data
    
    def calculate_technical_indicators(self):
        """
        Calculate technical indicators
        """
        if self.data is None:
            raise ValueError("Fetch stock data first")
        
        # Moving Averages
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA200'] = self.data['Close'].rolling(window=200).mean()
        
        # Relative Strength Index (RSI)
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        return self.data
    
    def predict_stock_price(self):
        """
        Machine Learning stock price prediction
        """
        if self.data is None:
            raise ValueError("Fetch stock data first")
        
        # Prepare features
        features = ['Open', 'High', 'Low', 'Volume']
        X = self.data[features]
        y = self.data['Close']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Predict
        self.predictions = {
            'train_score': rf_model.score(X_train_scaled, y_train),
            'test_score': rf_model.score(X_test_scaled, y_test),
            'future_predictions': rf_model.predict(X_test_scaled)
        }
        
        return self.predictions
    
    def visualize_stock_data(self):
        """
        Create interactive stock visualization
        """
        if self.data is None:
            raise ValueError("Fetch stock data first")
        
        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=self.data.index,
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close']
        )])
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=self.data.index, 
            y=self.data['MA50'], 
            mode='lines', 
            name='50-day MA'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data.index, 
            y=self.data['MA200'], 
            mode='lines', 
            name='200-day MA'
        ))
        
        fig.update_layout(
            title=f'{self.ticker} Stock Price Analysis',
            xaxis_title='Date',
            yaxis_title='Price'
        )
        
        return fig

def main():
    # Example usage
    stock_analyzer = StockMarketAnalytics(
        ticker='AAPL', 
        start_date='2022-01-01', 
        end_date='2024-01-01'
    )
    
    # Fetch and analyze data
    stock_data = stock_analyzer.fetch_stock_data()
    stock_analyzer.calculate_technical_indicators()
    predictions = stock_analyzer.predict_stock_price()
    
    # Visualize
    chart = stock_analyzer.visualize_stock_data()
    chart.show()

if __name__ == '__main__':
    main()
