#!/usr/bin/env python3
"""
Stock Market Prediction Script
This script makes predictions using trained models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import StockDataFetcher
from feature_engineering import FeatureEngineer

class StockPredictor:
    """Class for making stock predictions"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.load_models()
    
    def load_models(self):
        """Load trained models and scalers"""
        try:
            # Load feature columns
            if os.path.exists(f"{self.model_dir}/feature_columns.pkl"):
                self.feature_columns = joblib.load(f"{self.model_dir}/feature_columns.pkl")
                print(f"[INFO] Loaded {len(self.feature_columns)} feature columns")
            
            # Load scalers
            if os.path.exists(f"{self.model_dir}/scalers.pkl"):
                self.scalers = joblib.load(f"{self.model_dir}/scalers.pkl")
                print(f"[INFO] Loaded scalers: {list(self.scalers.keys())}")
            
            # Load traditional ML models
            model_files = {
                'linear_regression': 'linear_regression.pkl',
                'ridge': 'ridge.pkl',
                'lasso': 'lasso.pkl',
                'random_forest': 'random_forest.pkl',
                'gradient_boosting': 'gradient_boosting.pkl'
            }
            
            for name, filename in model_files.items():
                filepath = f"{self.model_dir}/{filename}"
                if os.path.exists(filepath):
                    self.models[name] = joblib.load(filepath)
                    print(f"[INFO] Loaded {name} model")
            
            # Load LSTM model
            lstm_path = f"{self.model_dir}/lstm_model.h5"
            if os.path.exists(lstm_path):
                self.models['lstm'] = load_model(lstm_path)
                print("[INFO] Loaded LSTM model")
            
            if not self.models:
                print("[WARNING] No trained models found. Please train models first.")
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
    
    def prepare_features(self, data):
        """
        Prepare features for prediction
        
        Args:
            data: Raw stock data
        
        Returns:
            Prepared features DataFrame
        """
        # Engineer features
        engineer = FeatureEngineer()
        featured_data = engineer.engineer_all_features(data, target_days=1)
        
        # Select only the features used in training
        if self.feature_columns:
            # Get the last row (most recent data) and select features
            latest_features = featured_data[self.feature_columns].iloc[-1:].dropna(axis=1)
            
            # Handle missing features
            missing_features = set(self.feature_columns) - set(latest_features.columns)
            if missing_features:
                print(f"[WARNING] Missing features: {missing_features}")
                # Fill missing features with 0
                for feature in missing_features:
                    latest_features[feature] = 0
            
            # Reorder columns to match training
            latest_features = latest_features[self.feature_columns]
            
            return latest_features
        else:
            print("[ERROR] No feature columns loaded")
            return None
    
    def make_predictions(self, features):
        """
        Make predictions using all available models
        
        Args:
            features: Prepared features DataFrame
        
        Returns:
            Dictionary of predictions
        """
        predictions = {}
        
        if features is None or features.empty:
            print("[ERROR] No features provided for prediction")
            return predictions
        
        # Scale features if scaler is available
        if 'standard' in self.scalers:
            features_scaled = self.scalers['standard'].transform(features)
        else:
            features_scaled = features.values
        
        # Make predictions with traditional ML models
        for name, model in self.models.items():
            if name == 'lstm':
                continue  # Handle LSTM separately
            
            try:
                pred = model.predict(features_scaled)[0]
                predictions[name] = pred
                print(f"[INFO] {name} prediction: ${pred:.2f}")
            except Exception as e:
                print(f"[ERROR] Failed to predict with {name}: {e}")
        
        # Handle LSTM prediction (requires sequence)
        if 'lstm' in self.models:
            try:
                # For LSTM, we need a sequence of data
                # This is a simplified version - in practice, you'd need the full sequence
                print("[INFO] LSTM prediction requires sequence data (not implemented in this simple version)")
            except Exception as e:
                print(f"[ERROR] Failed to predict with LSTM: {e}")
        
        return predictions
    
    def predict_stock(self, symbol, days_ahead=1):
        """
        Predict stock price for a given symbol
        
        Args:
            symbol: Stock symbol
            days_ahead: Number of days ahead to predict
        
        Returns:
            Dictionary of predictions
        """
        print(f"[INFO] Making predictions for {symbol}")
        
        # Fetch recent data
        fetcher = StockDataFetcher()
        data = fetcher.fetch_yahoo_data(symbol, period="1y")
        
        if data is None:
            print(f"[ERROR] Failed to fetch data for {symbol}")
            return {}
        
        # Get current price
        current_price = data['close'].iloc[-1]
        print(f"[INFO] Current price: ${current_price:.2f}")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Make predictions
        predictions = self.make_predictions(features)
        
        # Calculate prediction statistics
        if predictions:
            pred_values = list(predictions.values())
            avg_prediction = np.mean(pred_values)
            std_prediction = np.std(pred_values)
            
            print(f"\n[INFO] Prediction Summary:")
            print(f"Average prediction: ${avg_prediction:.2f}")
            print(f"Standard deviation: ${std_prediction:.2f}")
            print(f"Predicted change: ${avg_prediction - current_price:.2f} ({((avg_prediction - current_price) / current_price) * 100:.2f}%)")
            
            predictions['average'] = avg_prediction
            predictions['current_price'] = current_price
            predictions['predicted_change'] = avg_prediction - current_price
            predictions['predicted_change_pct'] = ((avg_prediction - current_price) / current_price) * 100
        
        return predictions
    
    def plot_predictions(self, symbol, predictions, save_plot=True):
        """
        Plot prediction results
        
        Args:
            symbol: Stock symbol
            predictions: Dictionary of predictions
            save_plot: Whether to save the plot
        """
        if not predictions:
            print("[WARNING] No predictions to plot")
            return
        
        # Extract model predictions (exclude metadata)
        model_preds = {k: v for k, v in predictions.items() 
                      if k not in ['average', 'current_price', 'predicted_change', 'predicted_change_pct']}
        
        if not model_preds:
            return
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot current price and predictions
        models = list(model_preds.keys())
        pred_values = list(model_preds.values())
        current_price = predictions.get('current_price', 0)
        
        plt.bar(models, pred_values, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.axhline(y=current_price, color='red', linestyle='--', label=f'Current Price: ${current_price:.2f}')
        
        if 'average' in predictions:
            plt.axhline(y=predictions['average'], color='green', linestyle='-', 
                       label=f'Average Prediction: ${predictions["average"]:.2f}')
        
        plt.title(f'{symbol} Stock Price Predictions')
        plt.xlabel('Models')
        plt.ylabel('Predicted Price ($)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'{symbol}_predictions.png', dpi=300, bbox_inches='tight')
            print(f"[INFO] Prediction plot saved as {symbol}_predictions.png")
        
        plt.show()
    
    def batch_predict(self, symbols, save_results=True):
        """
        Make predictions for multiple stocks
        
        Args:
            symbols: List of stock symbols
            save_results: Whether to save results to CSV
        
        Returns:
            DataFrame with all predictions
        """
        results = []
        
        for symbol in symbols:
            print(f"\n[INFO] Processing {symbol}...")
            predictions = self.predict_stock(symbol)
            
            if predictions:
                result = {'symbol': symbol}
                result.update(predictions)
                results.append(result)
        
        if results:
            results_df = pd.DataFrame(results)
            
            if save_results:
                filename = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                results_df.to_csv(filename, index=False)
                print(f"\n[INFO] Batch predictions saved to {filename}")
            
            return results_df
        
        return pd.DataFrame()

def main():
    """Main function for making predictions"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--days", type=int, default=1, help="Days ahead to predict")
    parser.add_argument("--batch", type=str, nargs='+', help="Multiple symbols for batch prediction")
    parser.add_argument("--plot", action="store_true", help="Plot predictions")
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = StockPredictor()
    
    if args.batch:
        # Batch prediction
        print(f"[INFO] Making batch predictions for: {args.batch}")
        results = predictor.batch_predict(args.batch)
        if not results.empty:
            print("\n[INFO] Batch Prediction Results:")
            print(results.to_string(index=False))
    else:
        # Single stock prediction
        predictions = predictor.predict_stock(args.symbol, args.days)
        
        if predictions and args.plot:
            predictor.plot_predictions(args.symbol, predictions)

if __name__ == "__main__":
    main()

