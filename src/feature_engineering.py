#!/usr/bin/env python3
"""
Feature Engineering for Stock Market Prediction
This module creates technical indicators and features for machine learning models.
"""

import pandas as pd
import numpy as np
import ta
from ta.utils import dropna

class FeatureEngineer:
    """Class to create technical indicators and features"""
    
    def __init__(self):
        pass
    
    def add_basic_features(self, df):
        """
        Add basic price-based features
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        
        # High-Low spread
        df['hl_spread'] = df['high'] - df['low']
        df['hl_spread_pct'] = (df['high'] - df['low']) / df['close']
        
        # Open-Close spread
        df['oc_spread'] = df['close'] - df['open']
        df['oc_spread_pct'] = (df['close'] - df['open']) / df['open']
        
        # Volume features
        df['volume_change'] = df['volume'].diff()
        df['volume_change_pct'] = df['volume'].pct_change()
        
        # Price position within day's range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def add_moving_averages(self, df, windows=[5, 10, 20, 50, 200]):
        """
        Add moving average indicators
        
        Args:
            df: DataFrame with price data
            windows: List of window sizes for moving averages
        
        Returns:
            DataFrame with moving average features
        """
        df = df.copy()
        
        for window in windows:
            # Simple Moving Average
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            
            # Exponential Moving Average
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
            # Price relative to moving averages
            df[f'close_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            df[f'close_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
        
        # Moving average crossovers
        if 20 in windows and 50 in windows:
            df['sma_20_50_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, 0)
            df['ema_20_50_cross'] = np.where(df['ema_20'] > df['ema_50'], 1, 0)
        
        return df
    
    def add_technical_indicators(self, df):
        """
        Add comprehensive technical indicators using ta library
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with technical indicators
        """
        df = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print("[ERROR] Missing required columns for technical indicators")
            return df
        
        try:
            # RSI (Relative Strength Index)
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Average True Range (ATR)
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Commodity Channel Index (CCI)
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            
            # Williams %R
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # On-Balance Volume (OBV)
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            # Money Flow Index (MFI)
            df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate some technical indicators: {e}")
        
        return df
    
    def add_lag_features(self, df, columns=['close', 'volume'], lags=[1, 2, 3, 5, 10]):
        """
        Add lagged features
        
        Args:
            df: DataFrame with data
            columns: Columns to create lags for
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_rolling_statistics(self, df, columns=['close', 'volume'], windows=[5, 10, 20]):
        """
        Add rolling statistical features
        
        Args:
            df: DataFrame with data
            columns: Columns to calculate statistics for
            windows: List of window sizes
        
        Returns:
            DataFrame with rolling statistics
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
                    df[f'{col}_max_{window}'] = df[col].rolling(window=window).max()
                    df[f'{col}_median_{window}'] = df[col].rolling(window=window).median()
        
        return df
    
    def add_time_features(self, df):
        """
        Add time-based features
        
        Args:
            df: DataFrame with datetime index
        
        Returns:
            DataFrame with time features
        """
        df = df.copy()
        
        # Extract time components
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Market session indicators
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    def create_target_variable(self, df, target_days=1, target_type='price'):
        """
        Create target variable for prediction
        
        Args:
            df: DataFrame with price data
            target_days: Number of days ahead to predict
            target_type: Type of target ('price', 'return', 'direction')
        
        Returns:
            DataFrame with target variable
        """
        df = df.copy()
        
        if target_type == 'price':
            df['target'] = df['close'].shift(-target_days)
        elif target_type == 'return':
            df['target'] = df['close'].pct_change(target_days).shift(-target_days)
        elif target_type == 'direction':
            future_price = df['close'].shift(-target_days)
            df['target'] = (future_price > df['close']).astype(int)
        
        return df
    
    def engineer_all_features(self, df, target_days=1, target_type='price'):
        """
        Apply all feature engineering steps
        
        Args:
            df: DataFrame with OHLCV data
            target_days: Number of days ahead to predict
            target_type: Type of target variable
        
        Returns:
            DataFrame with all engineered features
        """
        print("[INFO] Starting feature engineering...")
        
        # Clean data
        df = dropna(df)
        
        # Add all features
        df = self.add_basic_features(df)
        df = self.add_moving_averages(df)
        df = self.add_technical_indicators(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_statistics(df)
        df = self.add_time_features(df)
        df = self.create_target_variable(df, target_days, target_type)
        
        print(f"[INFO] Feature engineering complete. Shape: {df.shape}")
        print(f"[INFO] Features created: {len(df.columns)} columns")
        
        return df

def main():
    """Example usage of FeatureEngineer"""
    from data_fetcher import StockDataFetcher
    
    # Fetch sample data
    fetcher = StockDataFetcher()
    data = fetcher.fetch_yahoo_data("AAPL", period="1y")
    
    if data is not None:
        # Engineer features
        engineer = FeatureEngineer()
        featured_data = engineer.engineer_all_features(data)
        
        print(f"\nOriginal data shape: {data.shape}")
        print(f"Featured data shape: {featured_data.shape}")
        print(f"\nNew columns added: {len(featured_data.columns) - len(data.columns)}")
        
        # Show some features
        print(f"\nSample of engineered features:")
        feature_cols = [col for col in featured_data.columns if col not in data.columns]
        print(featured_data[feature_cols[:10]].head())

if __name__ == "__main__":
    main()

