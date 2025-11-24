import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import re
from datetime import datetime


class DataLoader:
    """Class to load and preprocess news data"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from file"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """Clean and preprocess the data"""
        if self.df is None:
            self.load_data()
        
        # Remove unnecessary columns
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis=1)
        
        # Convert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        
        # Remove rows with missing critical data
        self.df = self.df.dropna(subset=['headline', 'publisher', 'date'])
        
        # Clean text data
        self.df['headline_clean'] = self.df['headline'].apply(self._clean_text)
        self.df['headline_length'] = self.df['headline_clean'].str.len()
        
        # Extract time components
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['hour'] = self.df['date'].dt.hour
        
        print(f"Data preprocessing completed. Final shape: {self.df.shape}")
        return self.df
    
    def _clean_text(self, text: str) -> str:
        """Clean text data"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text


class EDAAnalyzer:
    """Class to perform Exploratory Data Analysis"""
    
    def __init__(self, data_loader: DataLoader):
        self.df = data_loader.df
        self.results = {}
    
    def descriptive_statistics(self) -> Dict[str, Any]:
        """Calculate descriptive statistics"""
        stats = {
            'total_articles': len(self.df),
            'date_range': {
                'start': self.df['date'].min(),
                'end': self.df['date'].max()
            },
            'headline_length_stats': {
                'mean': self.df['headline_length'].mean(),
                'median': self.df['headline_length'].median(),
                'std': self.df['headline_length'].std(),
                'min': self.df['headline_length'].min(),
                'max': self.df['headline_length'].max()
            },
            'publishers_count': self.df['publisher'].nunique(),
            'stocks_count': self.df['stock'].nunique()
        }
        self.results['descriptive_stats'] = stats
        return stats
    
    def publisher_analysis(self) -> pd.DataFrame:
        """Analyze publisher activity"""
        publisher_stats = self.df.groupby('publisher').agg({
            'headline': 'count',
            'headline_length': ['mean', 'std'],
            'stock': 'nunique'
        }).round(2)
        
        publisher_stats.columns = ['article_count', 'avg_headline_length',
                                 'std_headline_length', 'unique_stocks_covered']
        publisher_stats = publisher_stats.sort_values('article_count', ascending=False)
        
        self.results['publisher_analysis'] = publisher_stats
        return publisher_stats
    
    def time_series_analysis(self) -> Dict[str, pd.DataFrame]:
        """Analyze publication trends over time"""
        daily_counts = self.df.set_index('date').resample('D').size()
        weekly_counts = self.df.set_index('date').resample('W').size()
        monthly_counts = self.df.set_index('date').resample('M').size()

        
        time_analysis = {
            'daily': daily_counts,
            'weekly': weekly_counts,
            'monthly': monthly_counts,
            'day_of_week': self.df['day_of_week'].value_counts(),
            'hourly': self.df['hour'].value_counts().sort_index()
        }
        
        self.results['time_series_analysis'] = time_analysis
        return time_analysis