# src/sentiment_analysis.py
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class SimpleSentimentAnalyzer:
    """A simple rule-based sentiment analyzer as fallback"""
    
    def __init__(self):
        self.positive_words = {
            'good', 'great', 'excellent', 'positive', 'bullish', 'strong', 'growth',
            'profit', 'gain', 'rise', 'up', 'higher', 'success', 'win', 'benefit',
            'optimistic', 'boom', 'rally', 'surge', 'outperform', 'beat', 'positive'
        }
        self.negative_words = {
            'bad', 'poor', 'negative', 'bearish', 'weak', 'decline', 'loss', 'drop',
            'down', 'lower', 'failure', 'lose', 'risk', 'pessimistic', 'crash',
            'plunge', 'slump', 'underperform', 'miss', 'negative', 'warn'
        }
        self.intensifiers = {
            'very', 'extremely', 'highly', 'significantly', 'substantially'
        }
    
    def calculate_simple_sentiment(self, text):
        """Calculate sentiment using simple word counting"""
        if pd.isna(text) or text == "":
            return 0.0
        
        text_lower = str(text).lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_score = 0
        negative_score = 0
        
        for i, word in enumerate(words):
            if word in self.positive_words:
                # Check for intensifiers
                if i > 0 and words[i-1] in self.intensifiers:
                    positive_score += 2
                else:
                    positive_score += 1
            elif word in self.negative_words:
                if i > 0 and words[i-1] in self.intensifiers:
                    negative_score += 2
                else:
                    negative_score += 1
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        # Normalize score
        sentiment = (positive_score - negative_score) / max(total_words, 1)
        
        # Clip to reasonable range
        return max(min(sentiment, 1.0), -1.0)

class SentimentAnalyzer:
    def __init__(self):
        self.simple_analyzer = SimpleSentimentAnalyzer()
        self.advanced_available = False
        
        # Try to import advanced analyzers
        try:
            from textblob import TextBlob
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.TextBlob = TextBlob
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.advanced_available = True
            print("Advanced sentiment analysis (TextBlob + VADER) is available")
        except ImportError as e:
            print(f"Advanced sentiment analysis not available: {e}")
            print("Using simple rule-based sentiment analysis")
    
    def calculate_sentiment_scores(self, text):
        """Calculate sentiment scores using available methods"""
        if pd.isna(text) or text == "":
            return self._default_sentiment_scores()
        
        text_str = str(text)
        
        if self.advanced_available:
            return self._calculate_advanced_sentiment(text_str)
        else:
            return self._calculate_simple_sentiment(text_str)
    
    def _calculate_advanced_sentiment(self, text):
        """Calculate sentiment using TextBlob and VADER"""
        try:
            # TextBlob sentiment
            blob = self.TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            
            # VADER sentiment
            vader_scores = self.vader_analyzer.polarity_scores(text)
            vader_compound = vader_scores['compound']
            
            # Combined sentiment
            combined_score = (textblob_polarity + vader_compound) / 2
            
            sentiment_label = self._get_sentiment_label(combined_score)
            
            return {
                'textblob_polarity': textblob_polarity,
                'vader_compound': vader_compound,
                'combined_score': combined_score,
                'sentiment_label': sentiment_label
            }
        except Exception as e:
            print(f"Error in advanced sentiment analysis: {e}")
            return self._calculate_simple_sentiment(text)
    
    def _calculate_simple_sentiment(self, text):
        """Calculate sentiment using simple rule-based approach"""
        try:
            simple_score = self.simple_analyzer.calculate_simple_sentiment(text)
            sentiment_label = self._get_sentiment_label(simple_score)
            
            return {
                'textblob_polarity': simple_score,
                'vader_compound': simple_score,
                'combined_score': simple_score,
                'sentiment_label': sentiment_label
            }
        except Exception as e:
            print(f"Error in simple sentiment analysis: {e}")
            return self._default_sentiment_scores()
    
    def _get_sentiment_label(self, score):
        """Convert sentiment score to label"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _default_sentiment_scores(self):
        """Return default scores for empty/invalid text"""
        return {
            'textblob_polarity': 0,
            'vader_compound': 0,
            'combined_score': 0,
            'sentiment_label': 'neutral'
        }
    
    def analyze_news_sentiment(self, news_data):
        """Apply sentiment analysis to news data"""
        print("Applying sentiment analysis...")
        
        # Sample first few rows if dataset is large
        if len(news_data) > 1000:
            print(f"Large dataset detected ({len(news_data)} rows). Sampling first 1000 rows for demonstration.")
            sample_data = news_data.head(1000).copy()
        else:
            sample_data = news_data.copy()
        
        # Apply sentiment analysis
        sentiment_results = sample_data['clean_text'].apply(self.calculate_sentiment_scores)
        
        # Expand sentiment results into separate columns
        sentiment_df = pd.DataFrame(sentiment_results.tolist(), index=sample_data.index)
        news_data_with_sentiment = pd.concat([sample_data, sentiment_df], axis=1)
        
        print("Sentiment analysis complete!")
        print(f"Sentiment distribution: {news_data_with_sentiment['sentiment_label'].value_counts().to_dict()}")
        
        return news_data_with_sentiment

class CorrelationAnalyzer:
    def __init__(self, news_data, stock_data):
        self.news_data = news_data
        self.stock_data = stock_data
        
    def align_dates(self):
        """Align news and stock data by date"""
        print("Aligning news and stock data by date...")
        
        # Ensure both datasets have date information
        if 'date' not in self.news_data.columns:
            # Try to find date column
            date_columns = [col for col in self.news_data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                self.news_data['date'] = pd.to_datetime(self.news_data[date_columns[0]]).dt.date
            else:
                # Create dummy dates
                self.news_data['date'] = pd.date_range('2020-01-01', periods=len(self.news_data), freq='D').date
        
        # Ensure stock data has date column
        if not hasattr(self.stock_data.index, 'date'):
            self.stock_data['date'] = self.stock_data.index.date
        else:
            self.stock_data['date'] = self.stock_data.index.date
        
        # Aggregate daily sentiment scores
        daily_sentiment = self.news_data.groupby('date').agg({
            'combined_score': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: (x == 'positive').sum() / len(x)  # positive ratio
        }).round(4)
        
        # Flatten column names
        daily_sentiment.columns = ['_'.join(col).strip() if col[1] else col[0] for col in daily_sentiment.columns]
        daily_sentiment = daily_sentiment.rename(columns={
            'combined_score_mean': 'sentiment_mean',
            'combined_score_std': 'sentiment_std',
            'combined_score_count': 'article_count',
            'sentiment_label_<lambda>': 'positive_ratio'
        })
        
        # Calculate daily stock returns
        self.stock_data['daily_return'] = self.stock_data['Close'].pct_change()
        self.stock_data['next_day_return'] = self.stock_data['daily_return'].shift(-1)
        
        # Merge datasets
        merged_data = pd.merge(
            daily_sentiment,
            self.stock_data[['date', 'daily_return', 'next_day_return', 'Close']],
            on='date',
            how='inner'
        )
        
        # Remove rows with NaN values
        merged_data = merged_data.dropna()
        
        print(f"Merged dataset created with {len(merged_data)} days of data")
        return merged_data
    
    def calculate_correlations(self, merged_data):
        """Calculate Pearson correlations between sentiment and returns"""
        correlations = {}
        
        try:
            # Same-day correlations
            valid_data = merged_data[['sentiment_mean', 'daily_return']].dropna()
            if len(valid_data) > 2:
                corr, p_value = pearsonr(valid_data['sentiment_mean'], valid_data['daily_return'])
                correlations['same_day_sentiment_vs_return'] = (corr, p_value)
            else:
                correlations['same_day_sentiment_vs_return'] = (0, 1)
        except:
            correlations['same_day_sentiment_vs_return'] = (0, 1)
        
        try:
            # Next-day correlations
            valid_data = merged_data[['sentiment_mean', 'next_day_return']].dropna()
            if len(valid_data) > 2:
                corr, p_value = pearsonr(valid_data['sentiment_mean'], valid_data['next_day_return'])
                correlations['sentiment_vs_next_day_return'] = (corr, p_value)
            else:
                correlations['sentiment_vs_next_day_return'] = (0, 1)
        except:
            correlations['sentiment_vs_next_day_return'] = (0, 1)
        
        try:
            # Volume-based correlations
            valid_data = merged_data[['article_count', 'daily_return']].dropna()
            if len(valid_data) > 2:
                corr, p_value = pearsonr(valid_data['article_count'], valid_data['daily_return'])
                correlations['article_count_vs_return'] = (corr, p_value)
            else:
                correlations['article_count_vs_return'] = (0, 1)
        except:
            correlations['article_count_vs_return'] = (0, 1)
        
        try:
            # Positive ratio correlations
            valid_data = merged_data[['positive_ratio', 'daily_return']].dropna()
            if len(valid_data) > 2:
                corr, p_value = pearsonr(valid_data['positive_ratio'], valid_data['daily_return'])
                correlations['positive_ratio_vs_return'] = (corr, p_value)
            else:
                correlations['positive_ratio_vs_return'] = (0, 1)
        except:
            correlations['positive_ratio_vs_return'] = (0, 1)
        
        return correlations
    
    def visualize_correlations(self, merged_data):
        """Create correlation visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot: Sentiment vs Same-day Returns
        try:
            axes[0,0].scatter(merged_data['sentiment_mean'], merged_data['daily_return'], alpha=0.6)
            axes[0,0].set_xlabel('Daily Average Sentiment Score')
            axes[0,0].set_ylabel('Same-day Stock Return')
            axes[0,0].set_title('Sentiment vs Same-day Returns')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add correlation line if enough data
            if len(merged_data) > 2:
                z = np.polyfit(merged_data['sentiment_mean'], merged_data['daily_return'], 1)
                p = np.poly1d(z)
                axes[0,0].plot(merged_data['sentiment_mean'], p(merged_data['sentiment_mean']), "r--", alpha=0.8)
        except Exception as e:
            axes[0,0].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Sentiment vs Same-day Returns')
        
        # Scatter plot: Sentiment vs Next-day Returns
        try:
            axes[0,1].scatter(merged_data['sentiment_mean'], merged_data['next_day_return'], alpha=0.6)
            axes[0,1].set_xlabel('Daily Average Sentiment Score')
            axes[0,1].set_ylabel('Next-day Stock Return')
            axes[0,1].set_title('Sentiment vs Next-day Returns')
            axes[0,1].grid(True, alpha=0.3)
        except Exception as e:
            axes[0,1].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Sentiment vs Next-day Returns')
        
        # Time series: Sentiment and Returns
        try:
            if len(merged_data) > 0:
                ax2 = axes[1,0].twinx()
                line1 = axes[1,0].plot(merged_data.index, merged_data['sentiment_mean'],
                              color='blue', label='Sentiment Score')
                axes[1,0].set_ylabel('Sentiment Score', color='blue')
                axes[1,0].tick_params(axis='y', labelcolor='blue')
                
                line2 = ax2.plot(merged_data.index, merged_data['daily_return'],
                                color='red', label='Daily Return', alpha=0.7)
                ax2.set_ylabel('Daily Return', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                axes[1,0].legend(lines, labels, loc='upper left')
                
                axes[1,0].set_title('Time Series: Sentiment vs Returns')
            else:
                axes[1,0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[1,0].transAxes)
        except Exception as e:
            axes[1,0].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Time Series: Sentiment vs Returns')
        
        # Heatmap of correlations
        try:
            correlation_matrix = merged_data[['sentiment_mean', 'positive_ratio',
                                            'article_count', 'daily_return',
                                            'next_day_return']].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=axes[1,1], fmt='.3f')
            axes[1,1].set_title('Correlation Matrix')
        except Exception as e:
            axes[1,1].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        return fig