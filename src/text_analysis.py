import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from typing import List, Dict, Tuple
import re
from typing import Any

class TextAnalyzer:
    """Class for text analysis and topic modeling"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
        self.lda_model = None
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def preprocess_text(self, texts: List[str]) -> List[str]:
        """Preprocess text for analysis"""
        processed_texts = []
        for text in texts:
            # Tokenize
            tokens = word_tokenize(str(text).lower())
            # Remove stopwords and short tokens
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            # Lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            processed_texts.append(' '.join(tokens))
        return processed_texts
    
    def extract_keywords(self, texts: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
        """Extract most important keywords using TF-IDF"""
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        
        keywords = list(zip(feature_names, tfidf_scores))
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return keywords[:top_n]
    
    def topic_modeling_lda(self, texts: List[str], n_topics: int = 5) -> Tuple[Any, List[List[Tuple[str, float]]]]:
        """Perform LDA topic modeling"""
        processed_texts = self.preprocess_text(texts)
        
        # Vectorize using count vectorizer for LDA
        count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        count_matrix = count_vectorizer.fit_transform(processed_texts)
        
        # Apply LDA
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_output = self.lda_model.fit_transform(count_matrix)
        
        # Extract topics
        feature_names = count_vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_features_ind = topic.argsort()[:-10 - 1:-1]
            top_features = [(feature_names[i], topic[i]) for i in top_features_ind]
            topics.append(top_features)
        
        return self.lda_model, topics
    
    def analyze_publisher_keywords(self, df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """Analyze keywords by publisher"""
        publisher_keywords = {}
        for publisher in df['publisher'].unique():
            publisher_texts = df[df['publisher'] == publisher]['headline_clean'].tolist()
            if len(publisher_texts) > 10:
                keywords = self.extract_keywords(publisher_texts, top_n=15)
                publisher_keywords[publisher] = keywords
        
        return publisher_keywords