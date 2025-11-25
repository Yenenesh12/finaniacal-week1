import pytest
import pandas as pd
import numpy as np
from src.data_loader import DataLoader, EDAAnalyzer
from src.text_analyzer import TextAnalyzer

class TestEDA:
    """Test cases for EDA functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.sample_data = pd.DataFrame({
            'headline': ['Test headline one', 'Another test headline', 'Market news today'],
            'publisher': ['Publisher A', 'Publisher B', 'Publisher A'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'stock': ['AAPL', 'GOOGL', 'MSFT'],
            'url': ['http://test1.com', 'http://test2.com', 'http://test3.com']
        })
        
        # Save sample data for testing
        self.sample_data.to_csv('test_sample.csv', index=False)
    
    def test_data_loader(self):
        """Test DataLoader functionality"""
        loader = DataLoader('test_sample.csv')
        df = loader.load_data()
        
        assert df is not None
        assert len(df) == 3
        assert 'headline' in df.columns
    
    def test_data_preprocessing(self):
        """Test data preprocessing"""
        loader = DataLoader('test_sample.csv')
        df = loader.preprocess_data()
        
        assert 'headline_clean' in df.columns
        assert 'headline_length' in df.columns
        assert df['date'].dtype == 'datetime64[ns]'
    
    def test_eda_analyzer(self):
        """Test EDA analyzer"""
        loader = DataLoader('test_sample.csv')
        loader.preprocess_data()
        analyzer = EDAAnalyzer(loader)
        
        stats = analyzer.descriptive_statistics()
        assert stats['total_articles'] == 3
        assert 'publishers_count' in stats
    
    def test_text_analyzer(self):
        """Test text analyzer"""
        text_analyzer = TextAnalyzer()
        texts = ['This is a test headline', 'Another test for analysis']
        
        processed = text_analyzer.preprocess_text(texts)
        assert len(processed) == 2
        assert isinstance(processed[0], str)
    
    def teardown_method(self):
        """Cleanup test files"""
        import os
        if os.path.exists('test_sample.csv'):
            os.remove('test_sample.csv')