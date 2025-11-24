"""
Data analysis module for financial news analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FinancialNewsAnalyzer:
    """Class for analyzing financial news data."""
    
    def __init__(self, data_path=None):
        """
        Initialize the analyzer.
        
        Args:
            data_path (str): Path to the data file
        """
        self.data = None
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """
        Load data from CSV file.
        
        Args:
            data_path (str): Path to the data file
        """
        try:
            self.data = pd.read_csv(data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def descriptive_statistics(self):
        """Calculate descriptive statistics for textual data."""
        if self.data is None:
            print("No data loaded.")
            return
        
        stats = {}
        
        # Headline length analysis
        if 'headline' in self.data.columns:
            self.data['headline_length'] = self.data['headline'].str.len()
            stats['headline_length'] = {
                'mean': self.data['headline_length'].mean(),
                'std': self.data['headline_length'].std(),
                'min': self.data['headline_length'].min(),
                'max': self.data['headline_length'].max()
            }
        
        # Articles per publisher
        if 'publisher' in self.data.columns:
            publisher_counts = self.data['publisher'].value_counts()
            stats['publisher_counts'] = publisher_counts.head(10)  # Top 10 publishers
        
        return stats
    
    def time_series_analysis(self, date_column='publication_date'):
        """Analyze publication trends over time."""
        if self.data is None:
            print("No data loaded.")
            return
        
        if date_column not in self.data.columns:
            print(f"Date column '{date_column}' not found.")
            return
        
        # Convert to datetime
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Daily publication frequency
        daily_counts = self.data.set_index(date_column).resample('D').size()
        
        return daily_counts
    
    def text_analysis(self, text_column='headline'):
        """Perform text analysis on specified column."""
        if self.data is None:
            print("No data loaded.")
            return
        
        if text_column not in self.data.columns:
            print(f"Text column '{text_column}' not found.")
            return
        
        # Basic text preprocessing
        stop_words = set(stopwords.words('english'))
        financial_terms = set(['stock', 'market', 'price', 'share', 'trading', 'financial'])
        custom_stopwords = stop_words - financial_terms
        
        def preprocess_text(text):
            if pd.isna(text):
                return ""
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in custom_stopwords and len(token) > 2]
            return ' '.join(tokens)
        
        self.data['processed_text'] = self.data[text_column].apply(preprocess_text)
        
        # Extract common keywords
        all_text = ' '.join(self.data['processed_text'].tolist())
        words = word_tokenize(all_text)
        word_freq = Counter(words)
        
        return word_freq.most_common(20)
    
    def publisher_analysis(self):
        """Analyze publisher patterns."""
        if self.data is None:
            print("No data loaded.")
            return
        
        if 'publisher' not in self.data.columns:
            print("Publisher column not found.")
            return
        
        analysis = {}
        
        # Publisher frequency
        analysis['publisher_frequency'] = self.data['publisher'].value_counts()
        
        # Email domain analysis (if publishers are emails)
        email_publishers = self.data[self.data['publisher'].str.contains('@', na=False)]
        if not email_publishers.empty:
            email_publishers['domain'] = email_publishers['publisher'].str.split('@').str[1]
            analysis['domain_counts'] = email_publishers['domain'].value_counts()
        
        return analysis

def main():
    """Main function for demonstration."""
    print("Financial News Analysis Module")
    print("This module provides tools for analyzing financial news data.")

if __name__ == "__main__":
    main()