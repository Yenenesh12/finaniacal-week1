"""
Unit tests for data analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_analysis import FinancialNewsAnalyzer

class TestFinancialNewsAnalyzer:
    """Test cases for FinancialNewsAnalyzer class."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'headline': [
                'Stock market reaches all time high',
                'Federal Reserve announces interest rate decision',
                'Tech companies report earnings today',
                'Oil prices surge amid supply concerns',
                'Bank of America upgrades stock rating'
            ],
            'publisher': [
                'news@financialtimes.com',
                'alerts@bloomberg.com',
                'updates@reuters.com',
                'news@financialtimes.com',
                'alerts@bloomberg.com'
            ],
            'publication_date': [
                '2024-01-01',
                '2024-01-01',
                '2024-01-02',
                '2024-01-02',
                '2024-01-03'
            ]
        })
        
        self.analyzer = FinancialNewsAnalyzer()
        self.analyzer.data = self.sample_data
    
    def test_descriptive_statistics(self):
        """Test descriptive statistics calculation."""
        stats = self.analyzer.descriptive_statistics()
        
        assert 'headline_length' in stats
        assert 'publisher_counts' in stats
        assert stats['headline_length']['mean'] > 0
    
    def test_time_series_analysis(self):
        """Test time series analysis."""
        daily_counts = self.analyzer.time_series_analysis()
        
        assert len(daily_counts) > 0
        assert isinstance(daily_counts, pd.Series)
    
    def test_text_analysis(self):
        """Test text analysis functionality."""
        common_keywords = self.analyzer.text_analysis()
        
        assert len(common_keywords) <= 20
        assert isinstance(common_keywords, list)
    
    def test_publisher_analysis(self):
        """Test publisher analysis."""
        publisher_stats = self.analyzer.publisher_analysis()
        
        assert 'publisher_frequency' in publisher_stats
        assert 'domain_counts' in publisher_stats

def test_empty_analyzer():
    """Test analyzer without data."""
    analyzer = FinancialNewsAnalyzer()
    stats = analyzer.descriptive_statistics()
    assert stats is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])