"""
User Behavior Analysis Module for Amazon Reviews

This module analyzes user-based patterns including:
1. Superusers/Influencers identification
2. Review helpfulness patterns
3. User impact on product sales
4. Review quality indicators
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from scipy import stats
from ..models import AmazonReview


class UserBehaviorAnalyzer:
    """Analyzes user behavior patterns in Amazon review data."""
    
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
        df['text_length'] = df['text'].str.len()
        df['title_length'] = df['title'].str.len()
        df['is_positive'] = df['rating'] >= 4
        
        return df
    
    def identify_superusers_influencers(self, 
                                      helpful_vote_threshold: float = 0.8,
                                      min_reviews: int = 5) -> Dict[str, Any]:
        """
        Identify superusers and potential influencers based on helpful vote patterns.
        
        Args:
            helpful_vote_threshold: Percentile threshold for high helpful votes
            min_reviews: Minimum number of reviews to consider a user
            
        Returns:
            Dictionary with superuser analysis results
        """
        # Calculate user statistics
        user_stats = self.df.groupby('user_id').agg({
            'helpful_vote': ['count', 'sum', 'mean', 'std'],
            'rating': ['mean', 'std'],
            'verified_purchase': 'mean',
            'text_length': 'mean',
            'asin': 'nunique'  # Number of unique products reviewed
        }).round(3)
        
        user_stats.columns = [
            'review_count', 'total_helpful_votes', 'avg_helpful_votes', 'helpful_vote_std',
            'avg_rating', 'rating_std', 'verified_purchase_rate', 'avg_text_length',
            'unique_products'
        ]
        
        # Filter users with minimum review count
        active_users = user_stats[user_stats['review_count'] >= min_reviews].copy()
        
        if len(active_users) == 0:
            return {'message': 'No users meet minimum review criteria'}
        
        # Calculate helpful vote percentiles
        helpful_vote_percentile = np.percentile(
            active_users['avg_helpful_votes'], 
            helpful_vote_threshold * 100
        )
        
        # Identify potential superusers
        superusers = active_users[
            active_users['avg_helpful_votes'] >= helpful_vote_percentile
        ].copy()
        
        # Calculate influence metrics
        superusers['influence_score'] = (
            superusers['avg_helpful_votes'] * 0.4 +
            superusers['review_count'] * 0.3 +
            superusers['unique_products'] * 0.2 +
            superusers['verified_purchase_rate'] * 0.1
        )
        
        # Sort by influence score
        superusers = superusers.sort_values('influence_score', ascending=False)
        
        # Compare superusers to average users
        avg_user_stats = active_users.mean()
        superuser_comparison = {}
        
        for metric in ['avg_helpful_votes', 'review_count', 'avg_rating', 'verified_purchase_rate']:
            superuser_avg = superusers[metric].mean()
            population_avg = avg_user_stats[metric]
            superuser_comparison[metric] = {
                'superuser_avg': superuser_avg,
                'population_avg': population_avg,
                'multiplier': superuser_avg / population_avg if population_avg > 0 else 0
            }
        
        return {
            'total_active_users': len(active_users),
            'identified_superusers': len(superusers),
            'superuser_threshold': helpful_vote_percentile,
            'top_superusers': superusers.head(10).to_dict('index'),
            'superuser_comparison': superuser_comparison,
            'influence_distribution': {
                'mean': superusers['influence_score'].mean(),
                'std': superusers['influence_score'].std(),
                'top_score': superusers['influence_score'].max()
            }
        }
    
    def analyze_review_impact_on_sales(self, 
                                     time_window_days: int = 30) -> Dict[str, Any]:
        """
        Analyze correlation between influential reviews and subsequent review volume.
        
        Args:
            time_window_days: Days after review to measure impact
            
        Returns:
            Dictionary with impact analysis results
        """
        # Sort reviews by timestamp
        df_sorted = self.df.sort_values('timestamp')
        
        # Identify high-impact reviews (top 10% helpful votes)
        helpful_vote_threshold = self.df['helpful_vote'].quantile(0.9)
        high_impact_reviews = df_sorted[
            df_sorted['helpful_vote'] >= helpful_vote_threshold
        ].copy()
        
        impact_analysis = {}
        product_impacts = []
        
        for _, review in high_impact_reviews.iterrows():
            review_date = review['timestamp']
            product_asin = review['asin']
            
            # Get reviews for same product before and after this review
            product_reviews = df_sorted[df_sorted['asin'] == product_asin].copy()
            
            # Reviews before this review
            before_reviews = product_reviews[
                product_reviews['timestamp'] < review_date
            ]
            
            # Reviews after this review (within time window)
            after_cutoff = review_date + pd.Timedelta(days=time_window_days)
            after_reviews = product_reviews[
                (product_reviews['timestamp'] > review_date) &
                (product_reviews['timestamp'] <= after_cutoff)
            ]
            
            # Calculate impact metrics
            before_count = len(before_reviews)
            after_count = len(after_reviews)
            
            # Calculate review velocity (reviews per day)
            if len(before_reviews) > 0:
                before_days = (review_date - before_reviews['timestamp'].min()).days
                before_velocity = before_count / max(before_days, 1)
            else:
                before_velocity = 0
            
            after_velocity = after_count / time_window_days
            
            impact_data = {
                'review_id': f"{review['user_id']}_{product_asin}",
                'user_id': review['user_id'],
                'asin': product_asin,
                'helpful_votes': review['helpful_vote'],
                'rating': review['rating'],
                'review_date': review_date,
                'before_count': before_count,
                'after_count': after_count,
                'before_velocity': before_velocity,
                'after_velocity': after_velocity,
                'velocity_increase': after_velocity - before_velocity,
                'impact_ratio': after_velocity / before_velocity if before_velocity > 0 else 0
            }
            
            product_impacts.append(impact_data)
        
        # Convert to DataFrame for analysis
        impact_df = pd.DataFrame(product_impacts)
        
        if len(impact_df) > 0:
            # Statistical analysis
            velocity_correlation = stats.pearsonr(
                impact_df['helpful_votes'], 
                impact_df['velocity_increase']
            )
            
            # Top impactful reviews
            top_impacts = impact_df.nlargest(10, 'velocity_increase')
            
            impact_analysis = {
                'total_high_impact_reviews': len(impact_df),
                'avg_velocity_increase': impact_df['velocity_increase'].mean(),
                'correlation_helpful_votes_impact': {
                    'correlation': velocity_correlation[0],
                    'p_value': velocity_correlation[1]
                },
                'top_impactful_reviews': top_impacts.to_dict('records'),
                'impact_distribution': {
                    'positive_impacts': len(impact_df[impact_df['velocity_increase'] > 0]),
                    'negative_impacts': len(impact_df[impact_df['velocity_increase'] < 0]),
                    'no_impact': len(impact_df[impact_df['velocity_increase'] == 0])
                }
            }
        
        return impact_analysis
    
    def analyze_helpful_review_characteristics(self) -> Dict[str, Any]:
        """
        Analyze characteristics that make reviews more likely to be marked as helpful.
        
        Returns:
            Dictionary with helpful review analysis
        """
        # Define helpful review threshold (top 30%)
        helpful_threshold = self.df['helpful_vote'].quantile(0.7)
        self.df['is_helpful'] = self.df['helpful_vote'] >= helpful_threshold
        
        # Analyze characteristics
        characteristics = {}
        
        # Rating distribution for helpful vs non-helpful reviews
        rating_analysis = self.df.groupby(['is_helpful', 'rating']).size().unstack(fill_value=0)
        characteristics['rating_distribution'] = {
            'helpful_reviews': rating_analysis.loc[True].to_dict() if True in rating_analysis.index else {},
            'non_helpful_reviews': rating_analysis.loc[False].to_dict() if False in rating_analysis.index else {}
        }
        
        # Text length analysis
        helpful_reviews = self.df[self.df['is_helpful']]
        non_helpful_reviews = self.df[~self.df['is_helpful']]
        
        characteristics['text_length'] = {
            'helpful_avg': helpful_reviews['text_length'].mean(),
            'non_helpful_avg': non_helpful_reviews['text_length'].mean(),
            'helpful_median': helpful_reviews['text_length'].median(),
            'non_helpful_median': non_helpful_reviews['text_length'].median()
        }
        
        # Verified purchase impact
        verified_impact = self.df.groupby('is_helpful')['verified_purchase'].mean()
        characteristics['verified_purchase_impact'] = {
            'helpful_verified_rate': verified_impact.get(True, 0),
            'non_helpful_verified_rate': verified_impact.get(False, 0)
        }
        
        # Rating extremes analysis
        extreme_ratings = self.df[self.df['rating'].isin([1, 5])]
        moderate_ratings = self.df[self.df['rating'].isin([2, 3, 4])]
        
        characteristics['rating_extremes'] = {
            'extreme_helpful_rate': extreme_ratings['is_helpful'].mean() if len(extreme_ratings) > 0 else 0,
            'moderate_helpful_rate': moderate_ratings['is_helpful'].mean() if len(moderate_ratings) > 0 else 0
        }
        
        # Statistical tests
        from scipy.stats import ttest_ind, chi2_contingency
        
        # T-test for text length
        text_length_ttest = ttest_ind(
            helpful_reviews['text_length'].dropna(),
            non_helpful_reviews['text_length'].dropna()
        )
        
        characteristics['statistical_tests'] = {
            'text_length_difference': {
                't_statistic': text_length_ttest[0],
                'p_value': text_length_ttest[1],
                'significant': text_length_ttest[1] < 0.05
            }
        }
        
        return {
            'helpful_threshold': helpful_threshold,
            'total_helpful_reviews': len(helpful_reviews),
            'total_non_helpful_reviews': len(non_helpful_reviews),
            'characteristics': characteristics,
            'recommendations': self._generate_helpfulness_recommendations(characteristics)
        }
    
    def analyze_user_product_preferences(self, 
                                       preference_keywords: List[str] = None) -> Dict[str, Any]:
        """
        Analyze user requests for specific product features.
        
        Args:
            preference_keywords: List of keywords indicating user preferences
            
        Returns:
            Dictionary with preference analysis results
        """
        if preference_keywords is None:
            preference_keywords = [
                'larger size', 'bigger', 'smaller', 'travel size', 'mini',
                'organic', 'natural', 'chemical free', 'paraben free',
                'sensitive skin', 'hypoallergenic', 'fragrance free',
                'long lasting', 'waterproof', 'quick dry', 'non greasy'
            ]
        
        preference_analysis = {}
        
        # Combine title and text for keyword search
        all_text = (self.df['title'] + ' ' + self.df['text']).str.lower()
        
        for keyword in preference_keywords:
            # Find reviews mentioning this preference
            keyword_mask = all_text.str.contains(keyword, na=False)
            keyword_reviews = self.df[keyword_mask].copy()
            
            if len(keyword_reviews) > 0:
                preference_analysis[keyword] = {
                    'total_mentions': len(keyword_reviews),
                    'avg_rating': keyword_reviews['rating'].mean(),
                    'avg_helpful_votes': keyword_reviews['helpful_vote'].mean(),
                    'verified_purchase_rate': keyword_reviews['verified_purchase'].mean(),
                    'most_mentioned_products': keyword_reviews['asin'].value_counts().head(5).to_dict(),
                    'trend_over_time': self._analyze_keyword_trend(keyword_reviews)
                }
        
        # Overall preference insights
        total_preference_reviews = len(self.df[
            all_text.str.contains('|'.join(preference_keywords), na=False)
        ])
        
        return {
            'total_reviews_with_preferences': total_preference_reviews,
            'preference_rate': total_preference_reviews / len(self.df) if len(self.df) > 0 else 0,
            'keyword_analysis': preference_analysis,
            'top_requested_features': sorted(
                preference_analysis.items(),
                key=lambda x: x[1]['total_mentions'],
                reverse=True
            )[:5] if preference_analysis else []
        }
    
    def _analyze_keyword_trend(self, keyword_reviews: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend of keyword mentions over time."""
        if len(keyword_reviews) < 2:
            return {'trend': 'insufficient_data'}
        
        # Group by month
        monthly_counts = keyword_reviews.groupby(
            keyword_reviews['timestamp'].dt.to_period('M')
        ).size()
        
        if len(monthly_counts) < 2:
            return {'trend': 'insufficient_periods'}
        
        # Calculate trend
        x = np.arange(len(monthly_counts))
        y = monthly_counts.values
        slope, intercept, r_value, p_value, std_err = stats.linregr(x, y)
        
        return {
            'trend_slope': slope,
            'correlation': r_value,
            'p_value': p_value,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
            'monthly_counts': monthly_counts.to_dict()
        }
    
    def _generate_helpfulness_recommendations(self, characteristics: Dict[str, Any]) -> List[str]:
        """Generate recommendations for writing helpful reviews."""
        recommendations = []
        
        text_stats = characteristics.get('text_length', {})
        helpful_avg = text_stats.get('helpful_avg', 0)
        non_helpful_avg = text_stats.get('non_helpful_avg', 0)
        
        if helpful_avg > non_helpful_avg * 1.2:
            recommendations.append("Write longer, more detailed reviews for better helpfulness")
        
        verified_impact = characteristics.get('verified_purchase_impact', {})
        helpful_verified = verified_impact.get('helpful_verified_rate', 0)
        non_helpful_verified = verified_impact.get('non_helpful_verified_rate', 0)
        
        if helpful_verified > non_helpful_verified * 1.1:
            recommendations.append("Verified purchases tend to receive more helpful votes")
        
        rating_extremes = characteristics.get('rating_extremes', {})
        extreme_rate = rating_extremes.get('extreme_helpful_rate', 0)
        moderate_rate = rating_extremes.get('moderate_helpful_rate', 0)
        
        if extreme_rate > moderate_rate:
            recommendations.append("Strong opinions (1 or 5 stars) tend to be more helpful")
        
        return recommendations
