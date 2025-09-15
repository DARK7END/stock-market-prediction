#!/usr/bin/env python3
"""
Stock Market Analysis Tool
This script provides comprehensive stock analysis and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import StockDataFetcher
from feature_engineering import FeatureEngineer

class StockAnalyzer:
    """Class for comprehensive stock analysis"""
    
    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.engineer = FeatureEngineer()
    
    def basic_analysis(self, data, symbol):
        """
        Perform basic statistical analysis
        
        Args:
            data: Stock data DataFrame
            symbol: Stock symbol
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['symbol'] = symbol
        analysis['period_start'] = data.index.min().strftime('%Y-%m-%d')
        analysis['period_end'] = data.index.max().strftime('%Y-%m-%d')
        analysis['total_days'] = len(data)
        
        # Price statistics
        analysis['current_price'] = data['close'].iloc[-1]
        analysis['highest_price'] = data['high'].max()
        analysis['lowest_price'] = data['low'].min()
        analysis['average_price'] = data['close'].mean()
        analysis['price_volatility'] = data['close'].std()
        
        # Returns
        daily_returns = data['close'].pct_change().dropna()
        analysis['average_daily_return'] = daily_returns.mean()
        analysis['daily_return_volatility'] = daily_returns.std()
        analysis['total_return'] = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
        
        # Volume statistics
        analysis['average_volume'] = data['volume'].mean()
        analysis['max_volume'] = data['volume'].max()
        analysis['min_volume'] = data['volume'].min()
        
        # Risk metrics
        analysis['sharpe_ratio'] = analysis['average_daily_return'] / analysis['daily_return_volatility'] if analysis['daily_return_volatility'] > 0 else 0
        analysis['max_drawdown'] = self.calculate_max_drawdown(data['close'])
        
        return analysis
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def technical_analysis(self, data):
        """
        Perform technical analysis
        
        Args:
            data: Stock data DataFrame
        
        Returns:
            DataFrame with technical indicators
        """
        # Add technical indicators
        data_with_indicators = self.engineer.add_technical_indicators(data.copy())
        data_with_indicators = self.engineer.add_moving_averages(data_with_indicators)
        
        return data_with_indicators
    
    def plot_price_chart(self, data, symbol, save_plot=True):
        """
        Create interactive price chart with technical indicators
        
        Args:
            data: Stock data with technical indicators
            symbol: Stock symbol
            save_plot: Whether to save the plot
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Stock Price', 'Technical Indicators', 'Volume'),
            row_width=[0.2, 0.2, 0.7]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'sma_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['sma_20'], name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
        
        if 'sma_50' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['sma_50'], name='SMA 50', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Bollinger Bands
        if all(col in data.columns for col in ['bb_upper', 'bb_lower']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')),
                row=1, col=1
            )
        
        # RSI
        if 'rsi' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['rsi'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if all(col in data.columns for col in ['macd', 'macd_signal']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['macd'], name='MACD', line=dict(color='blue')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['macd_signal'], name='MACD Signal', line=dict(color='red')),
                row=2, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(x=data.index, y=data['volume'], name='Volume', marker_color='lightblue'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            xaxis_title='Date',
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Indicator Value", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        if save_plot:
            fig.write_html(f'{symbol}_analysis.html')
            print(f"[INFO] Interactive chart saved as {symbol}_analysis.html")
        
        fig.show()
    
    def plot_returns_analysis(self, data, symbol, save_plot=True):
        """
        Plot returns analysis
        
        Args:
            data: Stock data DataFrame
            symbol: Stock symbol
            save_plot: Whether to save the plot
        """
        # Calculate returns
        daily_returns = data['close'].pct_change().dropna()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{symbol} Returns Analysis', fontsize=16)
        
        # Daily returns time series
        axes[0, 0].plot(daily_returns.index, daily_returns * 100)
        axes[0, 0].set_title('Daily Returns (%)')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[0, 1].hist(daily_returns * 100, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Daily Return (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative returns
        cumulative_returns = (1 + daily_returns).cumprod()
        axes[1, 0].plot(cumulative_returns.index, cumulative_returns)
        axes[1, 0].set_title('Cumulative Returns')
        axes[1, 0].set_ylabel('Cumulative Return')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100
        axes[1, 1].plot(rolling_vol.index, rolling_vol)
        axes[1, 1].set_title('30-Day Rolling Volatility (%)')
        axes[1, 1].set_ylabel('Volatility (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'{symbol}_returns_analysis.png', dpi=300, bbox_inches='tight')
            print(f"[INFO] Returns analysis plot saved as {symbol}_returns_analysis.png")
        
        plt.show()
    
    def correlation_analysis(self, symbols, period="1y"):
        """
        Perform correlation analysis between multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Data period
        
        Returns:
            Correlation matrix DataFrame
        """
        print(f"[INFO] Performing correlation analysis for {len(symbols)} stocks")
        
        # Fetch data for all symbols
        stock_data = {}
        for symbol in symbols:
            data = self.fetcher.fetch_yahoo_data(symbol, period=period)
            if data is not None:
                stock_data[symbol] = data['close']
        
        if not stock_data:
            print("[ERROR] No data available for correlation analysis")
            return None
        
        # Create DataFrame with all closing prices
        prices_df = pd.DataFrame(stock_data)
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Stock Returns Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("[INFO] Correlation matrix plot saved as correlation_matrix.png")
        
        return correlation_matrix
    
    def generate_report(self, symbol, period="1y"):
        """
        Generate comprehensive analysis report
        
        Args:
            symbol: Stock symbol
            period: Data period
        
        Returns:
            Dictionary with complete analysis
        """
        print(f"[INFO] Generating comprehensive report for {symbol}")
        
        # Fetch data
        data = self.fetcher.fetch_yahoo_data(symbol, period=period)
        if data is None:
            print(f"[ERROR] Failed to fetch data for {symbol}")
            return None
        
        # Get stock info
        stock_info = self.fetcher.get_stock_info(symbol)
        
        # Basic analysis
        basic_stats = self.basic_analysis(data, symbol)
        
        # Technical analysis
        technical_data = self.technical_analysis(data)
        
        # Create visualizations
        self.plot_price_chart(technical_data, symbol)
        self.plot_returns_analysis(data, symbol)
        
        # Compile report
        report = {
            'stock_info': stock_info,
            'basic_analysis': basic_stats,
            'technical_data': technical_data,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print summary
        print(f"\n[INFO] Analysis Summary for {symbol}:")
        print("=" * 50)
        if stock_info:
            print(f"Company: {stock_info.get('company_name', 'N/A')}")
            print(f"Sector: {stock_info.get('sector', 'N/A')}")
        print(f"Current Price: ${basic_stats['current_price']:.2f}")
        print(f"Total Return: {basic_stats['total_return']:.2f}%")
        print(f"Volatility: {basic_stats['daily_return_volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {basic_stats['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {basic_stats['max_drawdown']:.2f}%")
        
        return report

def main():
    """Main function for stock analysis"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--period", type=str, default="1y", help="Data period")
    parser.add_argument("--correlation", type=str, nargs='+', help="Symbols for correlation analysis")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive report")
    args = parser.parse_args()
    
    analyzer = StockAnalyzer()
    
    if args.correlation:
        # Correlation analysis
        correlation_matrix = analyzer.correlation_analysis(args.correlation, args.period)
        if correlation_matrix is not None:
            print("\n[INFO] Correlation Matrix:")
            print(correlation_matrix.round(3))
    
    if args.report:
        # Generate comprehensive report
        report = analyzer.generate_report(args.symbol, args.period)
    else:
        # Basic analysis
        data = analyzer.fetcher.fetch_yahoo_data(args.symbol, args.period)
        if data is not None:
            basic_stats = analyzer.basic_analysis(data, args.symbol)
            print(f"\n[INFO] Basic Analysis for {args.symbol}:")
            for key, value in basic_stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

if __name__ == "__main__":
    main()

