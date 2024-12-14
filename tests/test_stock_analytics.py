import unittest
import pandas as pd
from datetime import datetime, timedelta
from src.stock_market_analytics import StockMarketAnalytics

class TestStockMarketAnalytics(unittest.TestCase):
    def setUp(self):
        self.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.analyzer = StockMarketAnalytics(ticker='AAPL', 
                                             start_date=self.start_date, 
                                             end_date=self.end_date)
    
    def test_fetch_stock_data(self):
        data = self.analyzer.fetch_stock_data()
        self.assertIsNotNone(data)
        self.assertTrue(len(data) > 0)
    
    def test_technical_indicators(self):
        self.analyzer.fetch_stock_data()
        indicators_data = self.analyzer.calculate_technical_indicators()
        
        self.assertTrue('MA50' in indicators_data.columns)
        self.assertTrue('MA200' in indicators_data.columns)
        self.assertTrue('RSI' in indicators_data.columns)
    
    def test_price_prediction(self):
        self.analyzer.fetch_stock_data()
        predictions = self.analyzer.predict_stock_price()
        
        self.assertIn('train_score', predictions)
        self.assertIn('test_score', predictions)
        self.assertIn('future_predictions', predictions)
    
    def test_visualization(self):
        self.analyzer.fetch_stock_data()
        self.analyzer.calculate_technical_indicators()
        chart = self.analyzer.visualize_stock_data()
        
        self.assertIsNotNone(chart)

if __name__ == '__main__':
    unittest.main()
