# 📊 Financial Analysis Project

A complete Python-based financial analysis toolkit that loads stock data, performs technical analysis, computes financial metrics, and visualizes market behavior.
This project uses **Yahoo Finance**, **TA-Lib**, **Pandas**, **NumPy**, and **Matplotlib** to analyze historical stock prices and generate insights.

---

## 🚀 Features

### **1. Data Loading**
- Fetch stock data using **Yahoo Finance API** (`yfinance`)
- Supports:
  - Period-based data (`1y`, `6mo`, `5y`, etc.)
  - Custom date ranges

### **2. Technical Indicators**
Using **TA-Lib**, the project computes:
- Moving Averages (SMA, EMA)
- RSI
- MACD & Histogram
- Stochastic Oscillator
- Bollinger Bands
- Average True Range (ATR)
- On-Balance Volume (OBV)
- Accumulation/Distribution (AD)

### **3. Financial Metrics**
- Daily returns
- Cumulative returns
- Rolling volatility
- Sharpe ratio
- Support & resistance levels
- Trend indicators (ADX, SAR)

### **4. Visualizations**
- Price + Moving Averages
- RSI Chart
- MACD Chart
- Volume Analysis
- Returns Analysis

All graphs are saved automatically to the **notebooks/** directory.

---

## 📁 Project Structure

├── .vscode/ # VS Code settings
├── .github/ # GitHub Actions workflows
├── src/ # Source code
├── notebooks/ # Jupyter notebooks for EDA
├── tests/ # Unit tests
└── scripts/ # Utility scripts

## Setup
1. Clone the repository
2. Create virtual environment: `python -m venv myenv`
3. Activate environment: `source venv/bin/activate` (Linux/Mac) or `myenv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Usage
- Run EDA: `jupyter notebook notebooks/exploratory_analysis.ipynb`
- Run tests: `pytest tests/`