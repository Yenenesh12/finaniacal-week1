# src/sentiment_analysis.py
import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SimpleSentimentAnalyzer:
    """A complete sentiment analyzer using only built-in Python libraries"""
    
    def __init__(self):
        # Comprehensive financial sentiment dictionaries
        self.positive_words = {
            # General positive
            'good', 'great', 'excellent', 'positive', 'strong', 'growth', 'profit',
            'gain', 'rise', 'up', 'higher', 'success', 'win', 'benefit', 'optimistic',
            'boom', 'rally', 'surge', 'outperform', 'beat', 'positive', 'bullish',
            'bull', 'soar', 'jump', 'climb', 'advance', 'improve', 'recovery',
            'rebound', 'thrive', 'flourish', 'prosper', 'expand', 'increase',
            'appreciate', 'strengthen', 'accelerate', 'momentum', 'breakout',
            # Financial specific
            'earnings', 'revenue', 'dividend', 'yield', 'upside', 'target',
            'upgrade', 'buy', 'outperform', 'overweight', 'strong buy'
        }
        
        self.negative_words = {
            # General negative
            'bad', 'poor', 'negative', 'weak', 'decline', 'loss', 'drop', 'down',
            'lower', 'failure', 'lose', 'risk', 'pessimistic', 'crash', 'plunge',
            'slump', 'underperform', 'miss', 'negative', 'warn', 'bearish', 'bear',
            'fall', 'slide', 'tumble', 'plummet', 'dip', 'retreat', 'weaken',
            'deteriorate', 'worsen', 'struggle', 'stagnate', 'shrink', 'contract',
            'decrease', 'depreciate', 'collapse', 'crisis', 'recession',
            # Financial specific
            'loss', 'bankruptcy', 'default', 'downgrade', 'sell', 'underperform',
            'underweight', 'short', 'volatility', 'uncertainty', 'risk'
        }
        
        self.intensifiers = {
            'very', 'extremely', 'highly', 'significantly', 'substantially',
            'dramatically', 'sharply', 'considerably', 'remarkably'
        }
        
        self.negators = {
            'not', 'no', 'never', 'none', 'nothing', 'without', 'lack'
        }
        
    def preprocess_text(self, text):
        """Clean and preprocess text for sentiment analysis"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def calculate_sentiment_score(self, text):
        """Calculate sentiment score using rule-based approach"""
        if not text or text == "":
            return 0.0
        
        text_clean = self.preprocess_text(text)
        words = re.findall(r'\b\w+\b', text_clean)
        
        if not words:
            return 0.0
        
        positive_score = 0
        negative_score = 0
        negation_active = False
        
        for i, word in enumerate(words):
            # Check for negators
            if word in self.negators:
                negation_active = True
                continue
            
            # Check for positive words
            if word in self.positive_words:
                score = 1
                # Check for intensifiers
                if i > 0 and words[i-1] in self.intensifiers:
                    score = 2
                # Apply negation
                if negation_active:
                    score = -score
                    negation_active = False
                positive_score += score
            
            # Check for negative words
            elif word in self.negative_words:
                score = -1
                # Check for intensifiers
                if i > 0 and words[i-1] in self.intensifiers:
                    score = -2
                # Apply negation (double negative becomes positive)
                if negation_active:
                    score = -score
                    negation_active = False
                negative_score += score
            
            # Reset negation after one word (simple approach)
            if word not in self.negators:
                negation_active = False
        
        total_score = positive_score + negative_score
        # Normalize by text length (with smoothing)
        normalized_score = total_score / (len(words) + 1)
        
        # Clip to reasonable range and apply sigmoid-like scaling
        final_score = np.tanh(normalized_score * 3)  # Scale and compress
        
        return float(final_score)
    
    def get_sentiment_label(self, score):
        """Convert sentiment score to categorical label"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'

class SentimentAnalyzer:
    """Main sentiment analyzer class - works without external dependencies"""
    
    def __init__(self):
        self.analyzer = SimpleSentimentAnalyzer()
        print("Simple sentiment analyzer initialized (no external dependencies required)")
    
    def calculate_sentiment_scores(self, text):
        """Calculate sentiment scores for a single text"""
        score = self.analyzer.calculate_sentiment_score(text)
        
        return {
            'sentiment_score': score,
            'sentiment_label': self.analyzer.get_sentiment_label(score),
            'combined_score': score  # For compatibility with existing code
        }
    
    def analyze_news_sentiment(self, news_data, text_column='clean_text'):
        """Apply sentiment analysis to entire news dataset"""
        print("Applying sentiment analysis to news data...")
        
        # Make a copy to avoid modifying original data
        news_with_sentiment = news_data.copy()
        
        # Ensure we have the text column
        if text_column not in news_with_sentiment.columns:
            print(f"Warning: Column '{text_column}' not found. Using first available text column.")
            text_columns = [col for col in news_with_sentiment.columns
                          if news_with_sentiment[col].dtype == 'object']
            if text_columns:
                text_column = text_columns[0]
                print(f"Using column '{text_column}' for sentiment analysis")
            else:
                raise ValueError("No text column found for sentiment analysis")
        
        # Apply sentiment analysis (sample if dataset is large)
        if len(news_with_sentiment) > 1000:
            print(f"Large dataset detected ({len(news_with_sentiment)} rows). Sampling first 1000 rows.")
            sample_data = news_with_sentiment.head(1000).copy()
        else:
            sample_data = news_with_sentiment
        
        print("Calculating sentiment scores...")
        sentiment_results = sample_data[text_column].apply(self.calculate_sentiment_scores)
        
        # Expand sentiment results into separate columns
        sentiment_df = pd.DataFrame(sentiment_results.tolist(), index=sample_data.index)
        result_data = pd.concat([sample_data, sentiment_df], axis=1)
        
        # Show sentiment distribution
        sentiment_counts = result_data['sentiment_label'].value_counts()
        print("\nSentiment Analysis Complete!")
        print("Sentiment Distribution:")
        for label, count in sentiment_counts.items():
            percentage = (count / len(result_data)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        print(f"\nAverage sentiment score: {result_data['sentiment_score'].mean():.3f}")
        
        return result_data

class CorrelationAnalyzer:
    """Analyze correlations between sentiment and stock returns"""
    
    def __init__(self, news_data, stock_data):
        self.news_data = news_data
        self.stock_data = stock_data
    
    def align_dates(self):
        """Align news sentiment data with stock price data by date"""
        print("Aligning news and stock data by date...")
        
        # Ensure news data has date information
        if 'date' not in self.news_data.columns:
            # Try to find date column in news data
            date_columns = [col for col in self.news_data.columns
                          if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                self.news_data['date'] = pd.to_datetime(self.news_data[date_col]).dt.date
                print(f"Using '{date_col}' as date column for news data")
            else:
                # Create sequential dates
                start_date = pd.to_datetime('2020-01-01').date()
                self.news_data['date'] = [start_date + pd.Timedelta(days=i)
                                        for i in range(len(self.news_data))]
                print("Created sequential dates for news data")
        
        # Ensure stock data has date information
        if 'date' not in self.stock_data.columns:
            if hasattr(self.stock_data.index, 'date'):
                self.stock_data['date'] = self.stock_data.index.date
            else:
                # Create dates for stock data
                start_date = pd.to_datetime('2020-01-01').date()
                self.stock_data['date'] = [start_date + pd.Timedelta(days=i)
                                         for i in range(len(self.stock_data))]
        
        # Aggregate daily sentiment scores from news data
        daily_sentiment = self.news_data.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: (x == 'positive').sum() / len(x)  # positive ratio
        }).round(4)
        
        # Flatten column names
        daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
        daily_sentiment = daily_sentiment.rename(columns={
            'sentiment_score_mean': 'sentiment_mean',
            'sentiment_score_std': 'sentiment_std',
            'sentiment_score_count': 'article_count',
            'sentiment_label_<lambda>': 'positive_ratio'
        })
        
        # Calculate daily stock returns
        self.stock_data = self.stock_data.sort_values('date')
        self.stock_data['daily_return'] = self.stock_data['Close'].pct_change()
        self.stock_data['next_day_return'] = self.stock_data['daily_return'].shift(-1)
        
        # Merge datasets on date
        merged_data = pd.merge(
            daily_sentiment,
            self.stock_data[['date', 'daily_return', 'next_day_return', 'Close']],
            on='date',
            how='inner'
        ).dropna()
        
        print(f"Merged dataset created with {len(merged_data)} days of aligned data")
        print(f"Date range: {merged_data['date'].min()} to {merged_data['date'].max()}")
        
        return merged_data
    
    def calculate_correlations(self, merged_data):
        """Calculate Pearson correlations between sentiment and stock returns"""
        print("Calculating correlations...")
        
        correlations = {}
        
        # Helper function to safely calculate correlation
        def safe_pearsonr(x, y):
            valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
            if len(valid_data) > 2:
                return pearsonr(valid_data['x'], valid_data['y'])
            else:
                return (0, 1)  # No correlation if insufficient data
        
        # Same-day correlation: sentiment vs returns
        corr, p_value = safe_pearsonr(merged_data['sentiment_mean'], merged_data['daily_return'])
        correlations['same_day_sentiment_vs_return'] = (corr, p_value)
        
        # Next-day correlation: sentiment today vs returns tomorrow
        corr, p_value = safe_pearsonr(merged_data['sentiment_mean'], merged_data['next_day_return'])
        correlations['sentiment_vs_next_day_return'] = (corr, p_value)
        
        # Article volume vs returns
        corr, p_value = safe_pearsonr(merged_data['article_count'], merged_data['daily_return'])
        correlations['article_count_vs_return'] = (corr, p_value)
        
        # Positive ratio vs returns
        corr, p_value = safe_pearsonr(merged_data['positive_ratio'], merged_data['daily_return'])
        correlations['positive_ratio_vs_return'] = (corr, p_value)
        
        return correlations
    
    def visualize_correlations(self, merged_data):
        """Create comprehensive correlation visualizations"""
        print("Creating correlation visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Sentiment vs Same-day Returns
        axes[0,0].scatter(merged_data['sentiment_mean'], merged_data['daily_return'],
                         alpha=0.6, color='blue')
        axes[0,0].set_xlabel('Daily Average Sentiment Score')
        axes[0,0].set_ylabel('Same-day Stock Return')
        axes[0,0].set_title('Sentiment vs Same-day Returns')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add trend line
        if len(merged_data) > 1:
            z = np.polyfit(merged_data['sentiment_mean'], merged_data['daily_return'], 1)
            p = np.poly1d(z)
            axes[0,0].plot(merged_data['sentiment_mean'], p(merged_data['sentiment_mean']),
                          "r--", alpha=0.8, label=f'Trend (r = {np.corrcoef(merged_data["sentiment_mean"], merged_data["daily_return"])[0,1]:.3f})')
            axes[0,0].legend()
        
        # Plot 2: Sentiment vs Next-day Returns
        axes[0,1].scatter(merged_data['sentiment_mean'], merged_data['next_day_return'],
                         alpha=0.6, color='green')
        axes[0,1].set_xlabel('Daily Average Sentiment Score')
        axes[0,1].set_ylabel('Next-day Stock Return')
        axes[0,1].set_title('Sentiment vs Next-day Returns')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Time series of sentiment and returns
        if len(merged_data) > 0:
            ax2 = axes[1,0].twinx()
            
            # Plot sentiment
            color = 'tab:blue'
            axes[1,0].plot(merged_data['date'], merged_data['sentiment_mean'],
                          color=color, label='Sentiment Score')
            axes[1,0].set_xlabel('Date')
            axes[1,0].set_ylabel('Sentiment Score', color=color)
            axes[1,0].tick_params(axis='y', labelcolor=color)
            
            # Plot returns
            color = 'tab:red'
            ax2.plot(merged_data['date'], merged_data['daily_return'],
                    color=color, label='Daily Return', alpha=0.7)
            ax2.set_ylabel('Daily Return', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            axes[1,0].set_title('Time Series: Sentiment vs Returns')
            axes[1,0].legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # Plot 4: Correlation heatmap
        correlation_cols = ['sentiment_mean', 'positive_ratio', 'article_count',
                          'daily_return', 'next_day_return']
        available_cols = [col for col in correlation_cols if col in merged_data.columns]
        
        if len(available_cols) > 1:
            corr_matrix = merged_data[available_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=axes[1,1], fmt='.3f', square=True)
            axes[1,1].set_title('Correlation Matrix')
        else:
            axes[1,1].text(0.5, 0.5, 'Insufficient data\nfor correlation matrix',
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        return fig

    def comprehensive_analysis(self, merged_data):
        """Perform comprehensive correlation analysis with multiple lags"""
        print("\n=== COMPREHENSIVE CORRELATION ANALYSIS ===")
        
        # Calculate correlations for multiple lags
        lags = range(0, 6)  # Same day to 5 days ahead
        lag_results = []
        
        for lag in lags:
            if lag == 0:
                returns_col = 'daily_return'
            else:
                returns_col = f'return_lag_{lag}'
                merged_data[returns_col] = merged_data['daily_return'].shift(-lag)
            
            valid_data = merged_data[['sentiment_mean', returns_col]].dropna()
            if len(valid_data) > 2:
                corr, p_value = pearsonr(valid_data['sentiment_mean'], valid_data[returns_col])
                lag_results.append((lag, corr, p_value, len(valid_data)))
            else:
                lag_results.append((lag, 0, 1, 0))
        
        # Print results
        print("Lag Correlations (Sentiment vs Future Returns):")
        print("Lag | Correlation | P-value  | Significance | Samples")
        print("-" * 55)
        for lag, corr, p_value, n in lag_results:
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{lag:3d} | {corr:10.4f}  | {p_value:7.4f}  | {significance:11} | {n:7d}")
        
        return lag_results