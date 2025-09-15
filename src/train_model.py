#!/usr/bin/env python3
"""
Stock Market Prediction Model Training
This script trains various machine learning models for stock price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import StockDataFetcher
from feature_engineering import FeatureEngineer

class StockPredictor:
    """Class for training stock prediction models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_data(self, df, target_col='target', test_size=0.2):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion of data for testing
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        # Separate features and target
        feature_cols = [col for col in df_clean.columns if col != target_col and col != 'symbol']
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Store feature columns
        self.feature_columns = feature_cols
        
        # Time series split (important for time series data)
        # Use the last test_size portion as test set
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"[INFO] Training set size: {len(X_train)}")
        print(f"[INFO] Test set size: {len(X_test)}")
        print(f"[INFO] Number of features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test, scaler_type='standard'):
        """
        Scale features for training
        
        Args:
            X_train: Training features
            X_test: Test features
            scaler_type: Type of scaler ('standard' or 'minmax')
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        if scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[scaler_type] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def train_linear_models(self, X_train, X_test, y_train, y_test):
        """Train linear regression models"""
        print("[INFO] Training linear models...")
        
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0)
        }
        
        for name, model in models.items():
            print(f"[INFO] Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"[INFO] {name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
            # Store model
            self.models[name] = model
            
            # Save model
            joblib.dump(model, f"{self.model_dir}/{name}.pkl")
    
    def train_ensemble_models(self, X_train, X_test, y_train, y_test):
        """Train ensemble models"""
        print("[INFO] Training ensemble models...")
        
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            print(f"[INFO] Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"[INFO] {name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
            # Store model
            self.models[name] = model
            
            # Save model
            joblib.dump(model, f"{self.model_dir}/{name}.pkl")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.plot_feature_importance(model, name)
    
    def create_lstm_sequences(self, data, sequence_length=60):
        """
        Create sequences for LSTM training
        
        Args:
            data: Scaled data array
            sequence_length: Length of input sequences
        
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def train_lstm_model(self, X_train, X_test, y_train, y_test, sequence_length=60):
        """Train LSTM model"""
        print("[INFO] Training LSTM model...")
        
        # Scale data for LSTM
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, 'minmax')
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_lstm_sequences(X_train_scaled, sequence_length)
        X_test_seq, y_test_seq = self.create_lstm_sequences(X_test_scaled, sequence_length)
        
        if len(X_train_seq) == 0:
            print("[WARNING] Not enough data for LSTM sequences")
            return
        
        # Reshape for LSTM (samples, time steps, features)
        X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2]))
        X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2]))
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train_seq.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            batch_size=32,
            epochs=50,
            validation_data=(X_test_seq, y_test_seq),
            verbose=1
        )
        
        # Predictions
        y_pred = model.predict(X_test_seq)
        
        # Metrics
        mse = mean_squared_error(y_test_seq, y_pred)
        mae = mean_absolute_error(y_test_seq, y_pred)
        r2 = r2_score(y_test_seq, y_pred)
        
        print(f"[INFO] LSTM - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Store model
        self.models['lstm'] = model
        
        # Save model
        model.save(f"{self.model_dir}/lstm_model.h5")
        
        # Plot training history
        self.plot_training_history(history)
    
    def plot_feature_importance(self, model, model_name):
        """Plot feature importance for tree-based models"""
        if not hasattr(model, 'feature_importances_'):
            return
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title(f'Top 20 Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f"{self.model_dir}/{model_name}_feature_importance.png")
        plt.close()
        
        print(f"[INFO] Feature importance plot saved for {model_name}")
    
    def plot_training_history(self, history):
        """Plot LSTM training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.model_dir}/lstm_training_history.png")
        plt.close()
        
        print("[INFO] LSTM training history plot saved")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n[INFO] Model Evaluation Summary:")
        print("=" * 50)
        
        results = []
        
        for name, model in self.models.items():
            if name == 'lstm':
                continue  # LSTM requires special handling
            
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'MSE': mse,
                'MAE': mae,
                'R2': r2
            })
        
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv(f"{self.model_dir}/model_evaluation.csv", index=False)
        
        return results_df

def main():
    """Main function to train stock prediction models"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--period", type=str, default="2y", help="Data period")
    parser.add_argument("--model", type=str, default="all", 
                       choices=["all", "linear", "ensemble", "lstm"],
                       help="Model type to train")
    parser.add_argument("--target_days", type=int, default=1, help="Days ahead to predict")
    args = parser.parse_args()
    
    print(f"[INFO] Training models for {args.symbol}")
    
    # Fetch data
    fetcher = StockDataFetcher()
    data = fetcher.fetch_yahoo_data(args.symbol, period=args.period)
    
    if data is None:
        print("[ERROR] Failed to fetch data")
        return
    
    # Engineer features
    engineer = FeatureEngineer()
    featured_data = engineer.engineer_all_features(data, target_days=args.target_days)
    
    # Initialize predictor
    predictor = StockPredictor()
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(featured_data)
    
    # Scale features
    X_train_scaled, X_test_scaled = predictor.scale_features(X_train, X_test)
    
    # Train models based on selection
    if args.model in ["all", "linear"]:
        predictor.train_linear_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    if args.model in ["all", "ensemble"]:
        predictor.train_ensemble_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    if args.model in ["all", "lstm"]:
        predictor.train_lstm_model(X_train, X_test, y_train, y_test)
    
    # Evaluate models
    if args.model != "lstm":
        predictor.evaluate_models(X_test_scaled, y_test)
    
    # Save feature columns and scaler
    joblib.dump(predictor.feature_columns, f"{predictor.model_dir}/feature_columns.pkl")
    joblib.dump(predictor.scalers, f"{predictor.model_dir}/scalers.pkl")
    
    print(f"\n[INFO] Training complete! Models saved in {predictor.model_dir}/")

if __name__ == "__main__":
    main()

