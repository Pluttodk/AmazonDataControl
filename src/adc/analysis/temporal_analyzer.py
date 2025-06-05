"""
Temporal Analysis Module for Amazon Reviews

This module analyzes time-based patterns in Amazon reviews including:
1. Spikes in positive reviews over time
2. Seasonal product shifts
3. Helpful vote increases over time
4. Keyword frequency changes over time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import re
from ..models import AmazonReview


class TemporalAnalyzer:
    """Analyzes temporal patterns in Amazon review data."""
    
    def __init__(self, reviews: List[AmazonReview]):
        """
        Initialize with review data.
        
        Args:
            reviews: List of AmazonReview objects
        """
        self.reviews = reviews
        self.df = self._reviews_to_dataframe()
        
    def _reviews_to_dataframe(self) -> pd.DataFrame:
        """Convert reviews to pandas DataFrame for easier analysis."""
        data = []
        for review in self.reviews:
            data.append({
                'rating': review.rating,
                'title': review.title,
                'text': review.text,
                'helpful_vote': review.helpful_vote,
                'asin': review.asin,
                'parent_asin': review.parent_asin,
                'user_id': review.user_id,
                'timestamp': review.timestamp,
                'verified_purchase': review.verified_purchase
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['is_positive'] = df['rating'] >= 4
        
        return df
    
    def analyze_positive_review_spikes(self, window_days: int = 30) -> Dict[str, Any]:
        """
        Analyze spikes in positive reviews over time.
        
        Args:
            window_days: Rolling window size in days
            
        Returns:
            Dictionary with spike analysis results
        """
        # Group by date and calculate positive review percentage
        daily_stats = self.df.groupby(self.df['timestamp'].dt.date).agg({
            'rating': ['count', 'mean'],
            'is_positive': ['sum', 'mean']
        }).round(3)
        
        daily_stats.columns = ['total_reviews', 'avg_rating', 'positive_count', 'positive_rate']
        daily_stats = daily_stats.reset_index()
        daily_stats['timestamp'] = pd.to_datetime(daily_stats['timestamp'])
        
        # Calculate rolling averages
        daily_stats['rolling_positive_rate'] = daily_stats['positive_rate'].rolling(
            window=window_days, min_periods=1
        ).mean()
        
        # Identify spikes (positive rate > 1.5 * rolling average)
        spike_threshold = 1.5
        daily_stats['is_spike'] = (
            daily_stats['positive_rate'] > 
            spike_threshold * daily_stats['rolling_positive_rate']
        )
        
        spikes = daily_stats[daily_stats['is_spike']].copy()
        
        # Check for recurring spikes
        recurring_spikes = self._find_recurring_patterns(spikes)
        
        return {
            'total_spike_days': len(spikes),
            'spike_threshold': spike_threshold,
            'avg_spike_positive_rate': spikes['positive_rate'].mean() if len(spikes) > 0 else 0,
            'recurring_patterns': recurring_spikes,
            'spike_details': spikes.to_dict('records'),
            'monthly_spike_frequency': spikes.groupby(spikes['timestamp'].dt.month).size().to_dict()
        }
    
    def analyze_seasonal_product_shifts(self) -> Dict[str, Any]:
        """
        Analyze seasonal shifts in product reviews (e.g., sunscreen in summer).
        
        Returns:
            Dictionary with seasonal analysis results
        """
        # Define seasonal keywords
        seasonal_keywords = {
            'summer': ['sunscreen', 'sun', 'beach', 'swimming', 'tan', 'uv', 'spf'],
            'winter': ['winter', 'cold', 'dry', 'moisturizer', 'heater', 'indoor'],
            'spring': ['spring', 'allergy', 'pollen', 'fresh', 'renewal'],
            'fall': ['fall', 'autumn', 'back to school', 'harvest', 'seasonal']
        }
        
        # Extract seasonal patterns by month
        monthly_analysis = {}
        
        for season, keywords in seasonal_keywords.items():
            pattern = '|'.join(keywords)
            
            # Find reviews containing seasonal keywords
            seasonal_mask = (
                self.df['title'].str.contains(pattern, case=False, na=False) |
                self.df['text'].str.contains(pattern, case=False, na=False)
            )
            
            seasonal_reviews = self.df[seasonal_mask].copy()
            
            if len(seasonal_reviews) > 0:
                monthly_dist = seasonal_reviews.groupby('month').size()
                peak_month = monthly_dist.idxmax()
                peak_count = monthly_dist.max()
                
                monthly_analysis[season] = {
                    'total_reviews': len(seasonal_reviews),
                    'peak_month': peak_month,
                    'peak_count': peak_count,
                    'monthly_distribution': monthly_dist.to_dict(),
                    'avg_rating': seasonal_reviews['rating'].mean(),
                    'seasonal_index': peak_count / (len(seasonal_reviews) / 12)  # Seasonality strength
                }
        
        return {
            'seasonal_patterns': monthly_analysis,
            'strongest_seasonal_trend': max(
                monthly_analysis.items(), 
                key=lambda x: x[1].get('seasonal_index', 0)
            ) if monthly_analysis else None
        }
    
    def analyze_helpful_vote_increases(self, time_periods: int = 4) -> Dict[str, Any]:
        """
        Analyze products with increased helpful votes over time.
        
        Args:
            time_periods: Number of time periods to divide data into
            
        Returns:
            Dictionary with helpful vote increase analysis
        """
        # Divide timeline into periods
        min_date = self.df['timestamp'].min()
        max_date = self.df['timestamp'].max()
        date_range = max_date - min_date
        period_length = date_range / time_periods
        
        period_analysis = {}
        product_trends = defaultdict(list)
        
        for period in range(time_periods):
            period_start = min_date + period * period_length
            period_end = min_date + (period + 1) * period_length
            
            period_data = self.df[
                (self.df['timestamp'] >= period_start) & 
                (self.df['timestamp'] < period_end)
            ]
            
            # Calculate helpful vote stats by product
            product_stats = period_data.groupby('asin').agg({
                'helpful_vote': ['mean', 'sum', 'count'],
                'rating': 'mean'
            }).round(3)
            
            product_stats.columns = ['avg_helpful', 'total_helpful', 'review_count', 'avg_rating']
            
            period_analysis[f'period_{period + 1}'] = {
                'date_range': f"{period_start.date()} to {period_end.date()}",
                'total_reviews': len(period_data),
                'avg_helpful_votes': period_data['helpful_vote'].mean(),
                'top_products': product_stats.nlargest(5, 'avg_helpful').to_dict('index')
            }
            
            # Track trends for each product
            for asin, stats in product_stats.iterrows():
                product_trends[asin].append({
                    'period': period + 1,
                    'avg_helpful': stats['avg_helpful'],
                    'total_helpful': stats['total_helpful'],
                    'review_count': stats['review_count']
                })
        
        # Identify products with increasing helpful vote trends
        trending_products = {}
        for asin, periods in product_trends.items():
            if len(periods) >= 2:
                # Calculate trend slope
                helpful_votes = [p['avg_helpful'] for p in periods]
                trend_slope = np.polyfit(range(len(helpful_votes)), helpful_votes, 1)[0]
                
                if trend_slope > 0.1:  # Positive trend threshold
                    trending_products[asin] = {
                        'trend_slope': trend_slope,
                        'periods': periods,
                        'improvement': helpful_votes[-1] - helpful_votes[0]
                    }
        
        return {
            'period_analysis': period_analysis,
            'trending_products': trending_products,
            'products_with_increases': len(trending_products),
            'strongest_trend': max(
                trending_products.items(),
                key=lambda x: x[1]['trend_slope']
            ) if trending_products else None
        }
    
    def analyze_keyword_frequency_changes(self, 
                                        bad_keywords: List[str] = None,
                                        time_periods: int = 4) -> Dict[str, Any]:
        """
        Analyze changes in keyword frequency over time.
        
        Args:
            bad_keywords: List of negative keywords to track
            time_periods: Number of time periods to analyze
            
        Returns:
            Dictionary with keyword trend analysis
        """
        if bad_keywords is None:
            bad_keywords = ['itching', 'rash', 'allergic', 'burn', 'irritation', 
                          'side effect', 'worse', 'terrible', 'awful', 'horrible']
        
        # Divide timeline into periods
        min_date = self.df['timestamp'].min()
        max_date = self.df['timestamp'].max()
        date_range = max_date - min_date
        period_length = date_range / time_periods
        
        keyword_trends = defaultdict(list)
        period_summaries = {}
        
        for period in range(time_periods):
            period_start = min_date + period * period_length
            period_end = min_date + (period + 1) * period_length
            
            period_data = self.df[
                (self.df['timestamp'] >= period_start) & 
                (self.df['timestamp'] < period_end)
            ]
            
            # Combine title and text for keyword analysis
            all_text = (period_data['title'] + ' ' + period_data['text']).str.lower()
            
            period_keyword_counts = {}
            for keyword in bad_keywords:
                count = all_text.str.contains(keyword, na=False).sum()
                frequency = count / len(period_data) if len(period_data) > 0 else 0
                
                keyword_trends[keyword].append({
                    'period': period + 1,
                    'count': count,
                    'frequency': frequency,
                    'total_reviews': len(period_data)
                })
                
                period_keyword_counts[keyword] = {
                    'count': count,
                    'frequency': frequency
                }
            
            period_summaries[f'period_{period + 1}'] = {
                'date_range': f"{period_start.date()} to {period_end.date()}",
                'total_reviews': len(period_data),
                'keyword_counts': period_keyword_counts,
                'most_frequent_bad_keyword': max(
                    period_keyword_counts.items(),
                    key=lambda x: x[1]['frequency']
                ) if period_keyword_counts else None
            }
        
        # Calculate trend slopes for each keyword
        trending_keywords = {}
        for keyword, periods in keyword_trends.items():
            if len(periods) >= 2:
                frequencies = [p['frequency'] for p in periods]
                trend_slope = np.polyfit(range(len(frequencies)), frequencies, 1)[0]
                
                trending_keywords[keyword] = {
                    'trend_slope': trend_slope,
                    'current_frequency': frequencies[-1],
                    'initial_frequency': frequencies[0],
                    'change': frequencies[-1] - frequencies[0],
                    'periods': periods
                }
        
        return {
            'period_summaries': period_summaries,
            'keyword_trends': trending_keywords,
            'increasing_negative_keywords': {
                k: v for k, v in trending_keywords.items() 
                if v['trend_slope'] > 0.001  # Positive trend threshold
            },
            'most_concerning_trend': max(
                trending_keywords.items(),
                key=lambda x: x[1]['trend_slope']
            ) if trending_keywords else None
        }
    
    def _find_recurring_patterns(self, spikes_df: pd.DataFrame) -> Dict[str, Any]:
        """Find recurring patterns in spike data."""
        if len(spikes_df) < 2:
            return {'recurring_found': False}
        
        # Check for monthly recurring patterns
        monthly_spikes = spikes_df.groupby(spikes_df['timestamp'].dt.month).size()
        recurring_months = monthly_spikes[monthly_spikes > 1].to_dict()
        
        # Check for seasonal patterns
        quarterly_spikes = spikes_df.groupby(spikes_df['timestamp'].dt.quarter).size()
        seasonal_pattern = quarterly_spikes.to_dict()
        
        return {
            'recurring_found': len(recurring_months) > 0,
            'recurring_months': recurring_months,
            'seasonal_pattern': seasonal_pattern,
            'most_frequent_spike_month': monthly_spikes.idxmax() if len(monthly_spikes) > 0 else None
        }
