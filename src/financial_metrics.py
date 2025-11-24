import pandas as pd
import numpy as np
import talib
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


class FinancialDataLoader:
    """Class to load and prepare financial data"""
    
    def __init__(self):
        self.data = None
    
    def load_stock_data(self, symbol: str, period: str = "1y",
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            
            if start_date and end_date:
                self.data = stock.history(start=start_date, end=end_date)
            else:
                self.data = stock.history(period=period)
            
            # Ensure required columns are present
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in self.data.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Available: {self.data.columns.tolist()}")
            
            print(f"Loaded data for {symbol}. Shape: {self.data.shape}")
            return self.data
        
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            raise
    
    def prepare_data(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare data for technical analysis"""
        if data is not None:
            self.data = data
        
        if self.data is None:
            raise ValueError("No data available. Please load data first.")
        
        # Clean data
        self.data = self.data.dropna()
        
        # Ensure numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self.data = self.data.dropna()
        return self.data


class TechnicalAnalyzer:
    """Class for technical analysis using TA-Lib"""
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_moving_averages(self, data: pd.DataFrame,
                                periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Calculate various moving averages"""
        close_prices = data['Close'].values
        
        for period in periods:
            if len(close_prices) >= period:
                data[f'SMA_{period}'] = talib.SMA(close_prices, timeperiod=period)
                data[f'EMA_{period}'] = talib.EMA(close_prices, timeperiod=period)
        
        self.indicators['moving_averages'] = [f'SMA_{p}' for p in periods] + [f'EMA_{p}' for p in periods]
        return data
    
    def calculate_oscillators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum oscillators"""
        high, low, close = data['High'].values, data['Low'].values, data['Close'].values
        
        # RSI
        data['RSI'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        data['MACD_Histogram'] = macd_hist
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close)
        data['Stoch_K'] = slowk
        data['Stoch_D'] = slowd
        
        # Williams %R
        data['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
        
        self.indicators['oscillators'] = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                                        'Stoch_K', 'Stoch_D', 'Williams_R']
        return data
    
    def calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        high, low, close = data['High'].values, data['Low'].values, data['Close'].values
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        data['BB_Upper'] = bb_upper
        data['BB_Middle'] = bb_middle
        data['BB_Lower'] = bb_lower
        
        # ATR (Average True Range)
        data['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        
        self.indicators['volatility'] = ['BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR']
        return data
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        close, volume = data['Close'].values, data['Volume'].values
        
        # OBV (On Balance Volume)
        data['OBV'] = talib.OBV(close, volume)
        
        # AD (Accumulation/Distribution)
        high, low, close = data['High'].values, data['Low'].values, data['Close'].values
        data['AD'] = talib.AD(high, low, close, volume)
        
        self.indicators['volume'] = ['OBV', 'AD']
        return data
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        data = self.calculate_moving_averages(data)
        data = self.calculate_oscillators(data)
        data = self.calculate_volatility_indicators(data)
        data = self.calculate_volume_indicators(data)
        
        return data


class PyNanceAnalyzer:
    """Class for financial metrics using PyNance concepts"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return metrics"""
        close_prices = data['Close']
        
        # Daily returns
        data['Daily_Return'] = close_prices.pct_change()
        
        # Cumulative returns
        data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()
        
        # Volatility (rolling 20-day)
        data['Volatility_20D'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # Sharpe ratio (annualized)
        risk_free_rate = 0.02
        excess_returns = data['Daily_Return'] - (risk_free_rate / 252)
        data['Sharpe_Ratio'] = (excess_returns.rolling(window=252).mean() * 252) / \
                              (data['Daily_Return'].rolling(window=252).std() * np.sqrt(252))
        
        self.metrics['returns'] = ['Daily_Return', 'Cumulative_Return', 'Volatility_20D', 'Sharpe_Ratio']
        return data
    
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate support and resistance levels"""
        data['Resistance'] = data['High'].rolling(window=window).max()
        data['Support'] = data['Low'].rolling(window=window).min()
        
        self.metrics['levels'] = ['Support', 'Resistance']
        return data
    
    def calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based indicators"""
        close_prices = data['Close'].values
        
        # ADX (Average Directional Index)
        data['ADX'] = talib.ADX(data['High'].values, data['Low'].values, close_prices)
        
        # Parabolic SAR
        data['SAR'] = talib.SAR(data['High'].values, data['Low'].values)
        
        self.metrics['trend'] = ['ADX', 'SAR']
        return data


class FinancialVisualizer:
    """Class for financial data visualization"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.fig_size = (15, 10)
    
    def plot_price_indicators(self, data: pd.DataFrame, symbol: str):
        """Plot price with key indicators"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price with moving averages and Bollinger Bands
        axes[0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
        if 'SMA_20' in data.columns:
            axes[0].plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
        if 'SMA_50' in data.columns:
            axes[0].plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
        if 'BB_Upper' in data.columns:
            axes[0].plot(data.index, data['BB_Upper'], label='BB Upper', linestyle='--', alpha=0.6)
            axes[0].plot(data.index, data['BB_Lower'], label='BB Lower', linestyle='--', alpha=0.6)
            axes[0].fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.1)
        
        axes[0].set_title(f'{symbol} - Price and Moving Averages')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'RSI' in data.columns:
            axes[1].plot(data.index, data['RSI'], label='RSI', color='purple')
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
            axes[1].set_title('Relative Strength Index (RSI)')
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # MACD
        if 'MACD' in data.columns:
            axes[2].plot(data.index, data['MACD'], label='MACD', color='blue')
            axes[2].plot(data.index, data['MACD_Signal'], label='Signal', color='red')
            axes[2].bar(data.index, data['MACD_Histogram'], label='Histogram', alpha=0.3)
            axes[2].set_title('MACD')
            axes[2].set_ylabel('MACD')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'notebooks/{symbol}_price_indicators.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_volume_analysis(self, data: pd.DataFrame, symbol: str):
        """Plot volume analysis"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Volume with OBV
        axes[0].bar(data.index, data['Volume'], alpha=0.7, label='Volume')
        axes[0].set_title(f'{symbol} - Trading Volume')
        axes[0].set_ylabel('Volume')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # OBV
        if 'OBV' in data.columns:
            axes[1].plot(data.index, data['OBV'], label='On Balance Volume', color='orange')
            axes[1].set_title('On Balance Volume (OBV)')
            axes[1].set_ylabel('OBV')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'notebooks/{symbol}_volume_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_returns_analysis(self, data: pd.DataFrame, symbol: str):
        """Plot returns analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Daily returns
        axes[0,0].hist(data['Daily_Return'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Distribution of Daily Returns')
        axes[0,0].set_xlabel('Daily Return')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        
        # Cumulative returns
        if 'Cumulative_Return' in data.columns:
            axes[0,1].plot(data.index, data['Cumulative_Return'], label='Cumulative Return')
            axes[0,1].set_title('Cumulative Returns')
            axes[0,1].set_ylabel('Cumulative Return')
            axes[0,1].grid(True, alpha=0.3)
        
        # Volatility
        if 'Volatility_20D' in data.columns:
            axes[1,0].plot(data.index, data['Volatility_20D'], label='20D Volatility', color='red')
            axes[1,0].set_title('Rolling 20-Day Volatility')
            axes[1,0].set_ylabel('Volatility')
            axes[1,0].grid(True, alpha=0.3)
        
        # Sharpe Ratio
        if 'Sharpe_Ratio' in data.columns:
            axes[1,1].plot(data.index, data['Sharpe_Ratio'], label='Sharpe Ratio', color='green')
            axes[1,1].set_title('Sharpe Ratio')
            axes[1,1].set_ylabel('Sharpe Ratio')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'notebooks/{symbol}_returns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()