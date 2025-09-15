# Stock Market Prediction

## Project Description

This project implements a machine learning system for stock market prediction and analysis. You can use any stock to analyze and predict future price movements using various technical indicators, historical data, and machine learning algorithms.

## Features

- Real-time stock data fetching
- Technical indicator analysis (Moving Averages, RSI, MACD, etc.)
- Multiple machine learning models for prediction
- Interactive data visualization
- Backtesting capabilities
- Risk assessment and portfolio optimization

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- yfinance
- plotly
- tensorflow/keras
- ta-lib (optional for advanced technical analysis)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd stock-market-prediction
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Stock Analysis
```bash
python src/stock_analyzer.py --symbol AAPL --period 1y
```

### Train Prediction Model
```bash
python src/train_model.py --symbol AAPL --model lstm
```

### Make Predictions
```bash
python src/predict.py --symbol AAPL --days 30
```

### Run Web Dashboard
```bash
python src/dashboard.py
```

## Project Structure

```
stock-market-prediction/
├── src/
│   ├── stock_analyzer.py
│   ├── data_fetcher.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   ├── dashboard.py
│   └── utils.py
├── models/
│   ├── lstm_model.h5
│   ├── random_forest.pkl
│   └── linear_regression.pkl
├── data/
│   └── historical_data/
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_comparison.ipynb
├── requirements.txt
└── README.md
```

## Supported Models

1. **LSTM (Long Short-Term Memory)**: Deep learning model for time series prediction
2. **Random Forest**: Ensemble method for robust predictions
3. **Linear Regression**: Simple baseline model
4. **Support Vector Regression**: Non-linear regression model
5. **ARIMA**: Traditional time series forecasting model

## Technical Indicators

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Stochastic Oscillator
- Volume indicators

## Task Submission Requirements

As per the project requirements, please complete the following submission tasks:

1. **Host the code on GitHub Repository (public)**: This repository should be made public on GitHub
2. **Record the code and output in a video**: Create a demonstration video showing the code and its output
3. **Post the video on YouTube**: Upload your demonstration video to YouTube
4. **Share links of code (GitHub) and video (YouTube) as a post on your LinkedIn profile**: Create a LinkedIn post with both links
5. **Create a LinkedIn post in Task Submission form when shared and tag Uneeq Interns**: Tag relevant accounts when sharing
6. **Submit the LinkedIn link in Task Submission Form when shared with you**: Provide the LinkedIn post link in the submission form

## Data Sources

- Yahoo Finance (via yfinance)
- Alpha Vantage API
- Quandl
- Custom CSV data import

## Risk Disclaimer

This project is for educational purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals before making investment choices.

## Contributing

Feel free to contribute to this project by submitting pull requests or reporting issues.

## License

This project is open source and available under the MIT License.

