#!/usr/bin/env python3
"""
Stock Data Fetcher
This module handles fetching stock data from various sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import time

class StockDataFetcher:
    """Class to fetch stock data from various sources"""
    
    def __init__(self):
        self.data_dir = "data/historical_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_yahoo_data(self, symbol, period="1y", interval="1d"):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pandas.DataFrame: Stock data with OHLCV columns
        """
        try:
            print(f"[INFO] Fetching {symbol} data for period {period}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"[WARNING] No data found for symbol {symbol}")
                return None
            
            # Clean column names
            data.columns = [col.replace(' ', '_').lower() for col in data.columns]
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Save to CSV
            filename = f"{self.data_dir}/{symbol}_{period}_{interval}.csv"
            data.to_csv(filename)
            print(f"[INFO] Data saved to {filename}")
            
            return data
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {symbol}: {e}")
            return None
    
    def fetch_multiple_stocks(self, symbols, period="1y", interval="1d"):
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
        
        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        stock_data = {}
        
        for symbol in symbols:
            print(f"[INFO] Processing {symbol}...")
            data = self.fetch_yahoo_data(symbol, period, interval)
            if data is not None:
                stock_data[symbol] = data
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        return stock_data
    
    def get_stock_info(self, symbol):
        """
        Get detailed stock information
        
        Args:
            symbol: Stock symbol
        
        Returns:
            dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            stock_info = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'beta': info.get('beta', 'N/A'),
                '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
                'current_price': info.get('currentPrice', 'N/A')
            }
            
            return stock_info
            
        except Exception as e:
            print(f"[ERROR] Failed to get info for {symbol}: {e}")
            return None
    
    def load_saved_data(self, symbol, period="1y", interval="1d"):
        """
        Load previously saved stock data
        
        Args:
            symbol: Stock symbol
            period: Data period
            interval: Data interval
        
        Returns:
            pandas.DataFrame: Loaded stock data
        """
        filename = f"{self.data_dir}/{symbol}_{period}_{interval}.csv"
        
        try:
            if os.path.exists(filename):
                data = pd.read_csv(filename, index_col=0, parse_dates=True)
                print(f"[INFO] Loaded data from {filename}")
                return data
            else:
                print(f"[WARNING] File {filename} not found")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to load data from {filename}: {e}")
            return None
    
    def get_sp500_symbols(self):
        """
        Get list of S&P 500 stock symbols
        
        Returns:
            list: List of S&P 500 symbols
        """
        try:
            # Fetch S&P 500 list from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            print(f"[INFO] Found {len(symbols)} S&P 500 symbols")
            return symbols
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch S&P 500 symbols: {e}")
            # Return some common symbols as fallback
            return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']

def main():
    """Example usage of StockDataFetcher"""
    fetcher = StockDataFetcher()
    
    # Example: Fetch Apple stock data
    symbol = "AAPL"
    data = fetcher.fetch_yahoo_data(symbol, period="1y")
    
    if data is not None:
        print(f"\n[INFO] Data shape: {data.shape}")
        print(f"[INFO] Date range: {data.index.min()} to {data.index.max()}")
        print(f"\n[INFO] First 5 rows:")
        print(data.head())
        
        # Get stock info
        info = fetcher.get_stock_info(symbol)
        if info:
            print(f"\n[INFO] Stock Information:")
            for key, value in info.items():
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()

