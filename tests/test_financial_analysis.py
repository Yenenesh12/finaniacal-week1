import pytest
import pandas as pd
import numpy as np
from src.financial_analyzer import (FinancialDataLoader, TechnicalAnalyzer,
                                  PyNanceAnalyzer, FinancialVisualizer)

class TestFinancialAnalysis:
    """Test cases for financial analysis functionality"""
    
    def setup_method(self):
        """Setup test data"""
        # Create sample stock data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(150, 250, 100),
            'Low': np.random.uniform(80, 150, 100),
            'Close': np.random.uniform(120, 180, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_technical_analyzer(self):
        """Test technical indicator calculations"""
        analyzer = TechnicalAnalyzer()
        data_with_indicators = analyzer.calculate_all_indicators(self.sample_data.copy())
        
        # Check if indicators are calculated
        assert 'RSI' in data_with_indicators.columns
        assert 'MACD' in data_with_indicators.columns
        assert 'SMA_20' in data_with_indicators.columns
    
    def test_pynance_analyzer(self):
        """Test PyNance metrics calculations"""
        analyzer = PyNanceAnalyzer()
        data_with_metrics = analyzer.calculate_returns(self.sample_data.copy())
        
        assert 'Daily_Return' in data_with_metrics.columns
        assert 'Cumulative_Return' in data_with_metrics.columns
    
    def test_financial_loader(self):
        """Test financial data loader"""
        loader = FinancialDataLoader()
        data = loader.prepare_data(self.sample_data.copy())
        
        assert data is not None
        assert len(data) == 100