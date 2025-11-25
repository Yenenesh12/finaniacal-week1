# src/data_processing.py
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

class FinancialDataProcessor:
    def __init__(self):
        self.news_data = None
        self.stock_data = None
        
    def load_news_data(self, filepath):
        """Load and preprocess financial news data"""
        try:
            self.news_data = pd.read_csv(filepath)
            print(f"Columns in dataset: {list(self.news_data.columns)}")
            print(f"Dataset shape: {self.news_data.shape}")
            
            # Auto-detect date column
            date_column = self._detect_date_column()
            if date_column:
                self.news_data[date_column] = pd.to_datetime(self.news_data[date_column])
                print(f"Using '{date_column}' as date column")
            else:
                # If no date column found, create a dummy one
                print("No date column found. Creating dummy dates...")
                self.news_data['date'] = pd.date_range(start='2020-01-01', periods=len(self.news_data), freq='D')
                date_column = 'date'
            
            # Auto-detect text column
            text_column = self._detect_text_column()
            if text_column:
                print(f"Using '{text_column}' as text column")
                self.news_data['clean_text'] = self.news_data[text_column].apply(self.clean_text)
            else:
                print("No text column found. Please check your data.")
                return None
            
            # Store the column names for reference
            self.date_column = date_column
            self.text_column = text_column
            
            return self.news_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _detect_date_column(self):
        """Auto-detect date column in the dataset"""
        date_keywords = ['date', 'time', 'published', 'timestamp', 'datetime', 'created']
        for col in self.news_data.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                return col
        return None
    
    def _detect_text_column(self):
        """Auto-detect text column in the dataset"""
        text_keywords = ['headline', 'title', 'text', 'content', 'summary', 'description', 'news']
        for col in self.news_data.columns:
            if any(keyword in col.lower() for keyword in text_keywords):
                return col
        # If no obvious text column, use the first string column
        for col in self.news_data.columns:
            if self.news_data[col].dtype == 'object':
                return col
        return None
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        # Remove special characters and digits, but keep basic punctuation for sentiment
        text = re.sub(r'[^\w\s\.\!\?]', '', str(text))
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def descriptive_analysis(self):
        """Perform descriptive analysis on news data"""
        if self.news_data is None:
            raise ValueError("No news data loaded")
        
        analysis = {
            'total_articles': len(self.news_data),
            'date_range': {
                'start': self.news_data[self.date_column].min(),
                'end': self.news_data[self.date_column].max()
            },
            'daily_volume': self.news_data.groupby(self.news_data[self.date_column].dt.date).size(),
            'text_length_stats': {
                'mean': self.news_data['clean_text'].str.len().mean(),
                'std': self.news_data['clean_text'].str.len().std(),
                'min': self.news_data['clean_text'].str.len().min(),
                'max': self.news_data['clean_text'].str.len().max()
            }
        }
        
        # Add publisher analysis if publisher column exists
        if 'publisher' in self.news_data.columns:
            analysis['publisher_counts'] = self.news_data['publisher'].value_counts()
        
        return analysis
    
    def topic_modeling(self, n_topics=5, max_features=1000):
        """Perform topic modeling using LDA"""
        # Filter out empty texts
        valid_texts = self.news_data['clean_text'].fillna('')
        valid_texts = valid_texts[valid_texts.str.len() > 10]
        
        if len(valid_texts) == 0:
            print("No valid texts for topic modeling")
            return [], np.array([])
        
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        X = vectorizer.fit_transform(valid_texts)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        
        # Get top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words
            })
        
        return topics, lda.transform(X)