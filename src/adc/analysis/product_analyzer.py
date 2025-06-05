"""
Product Analysis Module for Amazon Reviews

This module analyzes product-specific patterns including:
1. Ingredient correlation with positive reviews
2. Organic product sales trends
3. Product quality indicators
4. Feature-based analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, Counter
from scipy import stats
import re
from ..models import AmazonReview


class ProductAnalyzer:
    """Analyzes product-specific patterns in Amazon review data."""
    
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
        df['is_positive'] = df['rating'] >= 4
        df['is_negative'] = df['rating'] <= 2
        df['combined_text'] = df['title'] + ' ' + df['text']
        
        return df
    
    def analyze_ingredient_correlation(self, 
                                     ingredients: List[str] = None,
                                     correlation_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Analyze correlation between ingredients mentioned and positive reviews.
        
        Args:
            ingredients: List of ingredients to analyze
            correlation_threshold: Minimum correlation to consider significant
            
        Returns:
            Dictionary with ingredient correlation analysis
        """
        if ingredients is None:
            # Common skincare/cosmetic ingredients
            ingredients = [
                'hyaluronic acid', 'vitamin c', 'retinol', 'niacinamide', 'salicylic acid',
                'glycolic acid', 'ceramide', 'peptide', 'collagen', 'aloe vera',
                'tea tree', 'jojoba', 'argan oil', 'coconut oil', 'shea butter',
                'zinc oxide', 'titanium dioxide', 'oxybenzone', 'avobenzone'
            ]
        
        ingredient_analysis = {}
        correlation_results = {}
        
        # Analyze each ingredient
        for ingredient in ingredients:
            # Find reviews mentioning this ingredient
            ingredient_pattern = ingredient.replace(' ', r'\s+')
            ingredient_mask = self.df['combined_text'].str.contains(
                ingredient_pattern, case=False, na=False, regex=True
            )
            
            if ingredient_mask.sum() < 5:  # Skip if too few mentions
                continue
            
            ingredient_reviews = self.df[ingredient_mask].copy()
            non_ingredient_reviews = self.df[~ingredient_mask].copy()
            
            # Calculate statistics
            ingredient_stats = {
                'total_mentions': len(ingredient_reviews),
                'avg_rating': ingredient_reviews['rating'].mean(),
                'positive_rate': ingredient_reviews['is_positive'].mean(),
                'negative_rate': ingredient_reviews['is_negative'].mean(),
                'avg_helpful_votes': ingredient_reviews['helpful_vote'].mean(),
                'verified_purchase_rate': ingredient_reviews['verified_purchase'].mean()
            }
            
            # Compare with non-ingredient reviews
            comparison_stats = {
                'avg_rating': non_ingredient_reviews['rating'].mean(),
                'positive_rate': non_ingredient_reviews['is_positive'].mean(),
                'negative_rate': non_ingredient_reviews['is_negative'].mean(),
                'avg_helpful_votes': non_ingredient_reviews['helpful_vote'].mean()
            }
            
            # Calculate correlation
            rating_correlation = stats.pointbiserialr(
                ingredient_mask.astype(int), 
                self.df['rating']
            )
            
            helpful_correlation = stats.pointbiserialr(
                ingredient_mask.astype(int),
                self.df['helpful_vote']
            )
            
            ingredient_analysis[ingredient] = {
                'stats': ingredient_stats,
                'comparison': comparison_stats,
                'rating_correlation': {
                    'correlation': rating_correlation[0],
                    'p_value': rating_correlation[1],
                    'significant': rating_correlation[1] < 0.05
                },
                'helpful_vote_correlation': {
                    'correlation': helpful_correlation[0],
                    'p_value': helpful_correlation[1],
                    'significant': helpful_correlation[1] < 0.05
                },
                'effect_size': {
                    'rating_difference': ingredient_stats['avg_rating'] - comparison_stats['avg_rating'],
                    'positive_rate_difference': ingredient_stats['positive_rate'] - comparison_stats['positive_rate']
                }
            }
            
            # Store significant correlations
            if abs(rating_correlation[0]) >= correlation_threshold and rating_correlation[1] < 0.05:
                correlation_results[ingredient] = rating_correlation[0]
        
        # Rank ingredients by positive correlation
        positive_correlations = {
            k: v for k, v in correlation_results.items() if v > 0
        }
        negative_correlations = {
            k: v for k, v in correlation_results.items() if v < 0
        }
        
        return {
            'total_ingredients_analyzed': len(ingredient_analysis),
            'significant_correlations': len(correlation_results),
            'ingredient_analysis': ingredient_analysis,
            'top_positive_ingredients': sorted(
                positive_correlations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'top_negative_ingredients': sorted(
                negative_correlations.items(), 
                key=lambda x: x[1]
            )[:5],
            'correlation_summary': {
                'positive_correlations': len(positive_correlations),
                'negative_correlations': len(negative_correlations),
                'strongest_positive': max(positive_correlations.items(), key=lambda x: x[1]) if positive_correlations else None,
                'strongest_negative': min(negative_correlations.items(), key=lambda x: x[1]) if negative_correlations else None
            }
        }
    
    def analyze_organic_product_trends(self, 
                                     time_periods: int = 4) -> Dict[str, Any]:
        """
        Analyze trends in organic product reviews and sales.
        
        Args:
            time_periods: Number of time periods to analyze
            
        Returns:
            Dictionary with organic product trend analysis
        """
        # Define organic/natural keywords
        organic_keywords = [
            'organic', 'natural', 'chemical free', 'paraben free', 'sulfate free',
            'cruelty free', 'vegan', 'plant based', 'botanical', 'herbal',
            'non toxic', 'clean beauty', 'green', 'eco friendly'
        ]
        
        # Create organic product indicator
        organic_pattern = '|'.join(organic_keywords)
        self.df['is_organic_mentioned'] = self.df['combined_text'].str.contains(
            organic_pattern, case=False, na=False
        )
        
        # Divide timeline into periods
        min_date = self.df['timestamp'].min()
        max_date = self.df['timestamp'].max()
        date_range = max_date - min_date
        period_length = date_range / time_periods
        
        period_analysis = {}
        organic_trends = []
        
        for period in range(time_periods):
            period_start = min_date + period * period_length
            period_end = min_date + (period + 1) * period_length
            
            period_data = self.df[
                (self.df['timestamp'] >= period_start) & 
                (self.df['timestamp'] < period_end)
            ]
            
            if len(period_data) == 0:
                continue
            
            # Organic product statistics for this period
            organic_reviews = period_data[period_data['is_organic_mentioned']]
            non_organic_reviews = period_data[~period_data['is_organic_mentioned']]
            
            organic_stats = {
                'total_reviews': len(period_data),
                'organic_reviews': len(organic_reviews),
                'organic_percentage': len(organic_reviews) / len(period_data) * 100,
                'organic_avg_rating': organic_reviews['rating'].mean() if len(organic_reviews) > 0 else 0,
                'non_organic_avg_rating': non_organic_reviews['rating'].mean() if len(non_organic_reviews) > 0 else 0,
                'organic_helpful_votes': organic_reviews['helpful_vote'].mean() if len(organic_reviews) > 0 else 0,
                'unique_organic_products': organic_reviews['asin'].nunique() if len(organic_reviews) > 0 else 0
            }
            
            period_analysis[f'period_{period + 1}'] = {
                'date_range': f"{period_start.date()} to {period_end.date()}",
                'stats': organic_stats
            }
            
            organic_trends.append({
                'period': period + 1,
                'organic_percentage': organic_stats['organic_percentage'],
                'organic_avg_rating': organic_stats['organic_avg_rating'],
                'unique_products': organic_stats['unique_organic_products']
            })
        
        # Calculate trends
        if len(organic_trends) >= 2:
            organic_percentages = [t['organic_percentage'] for t in organic_trends]
            rating_trends = [t['organic_avg_rating'] for t in organic_trends if t['organic_avg_rating'] > 0]
            product_counts = [t['unique_products'] for t in organic_trends]
            
            # Trend analysis
            percentage_trend = np.polyfit(range(len(organic_percentages)), organic_percentages, 1)[0]
            product_trend = np.polyfit(range(len(product_counts)), product_counts, 1)[0]
            
            trend_analysis = {
                'organic_percentage_trend': {
                    'slope': percentage_trend,
                    'direction': 'increasing' if percentage_trend > 0 else 'decreasing',
                    'change': organic_percentages[-1] - organic_percentages[0]
                },
                'product_variety_trend': {
                    'slope': product_trend,
                    'direction': 'increasing' if product_trend > 0 else 'decreasing',
                    'change': product_counts[-1] - product_counts[0]
                }
            }
        else:
            trend_analysis = {'message': 'Insufficient data for trend analysis'}
        
        # Overall organic vs non-organic comparison
        overall_organic = self.df[self.df['is_organic_mentioned']]
        overall_non_organic = self.df[~self.df['is_organic_mentioned']]
        
        overall_comparison = {
            'organic_count': len(overall_organic),
            'non_organic_count': len(overall_non_organic),
            'organic_percentage': len(overall_organic) / len(self.df) * 100,
            'rating_comparison': {
                'organic_avg': overall_organic['rating'].mean() if len(overall_organic) > 0 else 0,
                'non_organic_avg': overall_non_organic['rating'].mean() if len(overall_non_organic) > 0 else 0
            },
            'helpful_vote_comparison': {
                'organic_avg': overall_organic['helpful_vote'].mean() if len(overall_organic) > 0 else 0,
                'non_organic_avg': overall_non_organic['helpful_vote'].mean() if len(overall_non_organic) > 0 else 0
            }
        }
        
        return {
            'period_analysis': period_analysis,
            'trend_analysis': trend_analysis,
            'overall_comparison': overall_comparison,
            'keyword_frequency': self._analyze_organic_keywords(organic_keywords),
            'top_organic_products': overall_organic['asin'].value_counts().head(10).to_dict() if len(overall_organic) > 0 else {}
        }
    
    def analyze_product_quality_indicators(self) -> Dict[str, Any]:
        """
        Analyze indicators of product quality based on review patterns.
        
        Returns:
            Dictionary with quality indicator analysis
        """
        # Group by product (ASIN)
        product_stats = self.df.groupby('asin').agg({
            'rating': ['count', 'mean', 'std'],
            'helpful_vote': ['sum', 'mean'],
            'verified_purchase': 'mean',
            'is_positive': 'mean',
            'is_negative': 'mean',
            'text_length': 'mean'
        }).round(3)
        
        product_stats.columns = [
            'review_count', 'avg_rating', 'rating_std',
            'total_helpful_votes', 'avg_helpful_votes',
            'verified_rate', 'positive_rate', 'negative_rate', 'avg_text_length'
        ]
        
        # Filter products with minimum review count
        min_reviews = 3
        qualified_products = product_stats[product_stats['review_count'] >= min_reviews].copy()
        
        if len(qualified_products) == 0:
            return {'message': 'No products meet minimum review criteria'}
        
        # Calculate quality scores
        qualified_products['quality_score'] = (
            qualified_products['avg_rating'] * 0.3 +
            qualified_products['positive_rate'] * 0.2 +
            (1 - qualified_products['negative_rate']) * 0.2 +
            np.clip(qualified_products['avg_helpful_votes'] / 10, 0, 1) * 0.15 +
            qualified_products['verified_rate'] * 0.1 +
            np.clip((1 - qualified_products['rating_std'] / 2), 0, 1) * 0.05  # Lower std = more consistent
        )
        
        # Categorize products
        quality_threshold_high = qualified_products['quality_score'].quantile(0.8)
        quality_threshold_low = qualified_products['quality_score'].quantile(0.2)
        
        high_quality = qualified_products[qualified_products['quality_score'] >= quality_threshold_high]
        low_quality = qualified_products[qualified_products['quality_score'] <= quality_threshold_low]
        medium_quality = qualified_products[
            (qualified_products['quality_score'] > quality_threshold_low) &
            (qualified_products['quality_score'] < quality_threshold_high)
        ]
        
        # Analyze quality patterns
        quality_patterns = {
            'high_quality': {
                'count': len(high_quality),
                'avg_rating': high_quality['avg_rating'].mean(),
                'avg_helpful_votes': high_quality['avg_helpful_votes'].mean(),
                'avg_verified_rate': high_quality['verified_rate'].mean(),
                'products': high_quality.sort_values('quality_score', ascending=False).head(5).to_dict('index')
            },
            'low_quality': {
                'count': len(low_quality),
                'avg_rating': low_quality['avg_rating'].mean(),
                'avg_helpful_votes': low_quality['avg_helpful_votes'].mean(),
                'avg_verified_rate': low_quality['verified_rate'].mean(),
                'products': low_quality.sort_values('quality_score').head(5).to_dict('index')
            },
            'medium_quality': {
                'count': len(medium_quality),
                'avg_rating': medium_quality['avg_rating'].mean(),
                'avg_helpful_votes': medium_quality['avg_helpful_votes'].mean(),
                'avg_verified_rate': medium_quality['verified_rate'].mean()
            }
        }
        
        # Quality indicators
        quality_indicators = self._identify_quality_indicators(qualified_products)
        
        return {
            'total_products_analyzed': len(qualified_products),
            'quality_distribution': {
                'high_quality': len(high_quality),
                'medium_quality': len(medium_quality),
                'low_quality': len(low_quality)
            },
            'quality_patterns': quality_patterns,
            'quality_indicators': quality_indicators,
            'quality_score_stats': {
                'mean': qualified_products['quality_score'].mean(),
                'std': qualified_products['quality_score'].std(),
                'min': qualified_products['quality_score'].min(),
                'max': qualified_products['quality_score'].max()
            }
        }
    
    def _analyze_organic_keywords(self, keywords: List[str]) -> Dict[str, int]:
        """Analyze frequency of organic keywords in reviews."""
        keyword_counts = {}
        
        for keyword in keywords:
            count = self.df['combined_text'].str.contains(
                keyword, case=False, na=False
            ).sum()
            keyword_counts[keyword] = count
        
        return dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _identify_quality_indicators(self, product_stats: pd.DataFrame) -> Dict[str, Any]:
        """Identify key indicators of product quality."""
        correlations = {}
        
        # Calculate correlations between different metrics and quality score
        metrics = ['avg_rating', 'avg_helpful_votes', 'verified_rate', 'positive_rate', 'rating_std']
        
        for metric in metrics:
            correlation = stats.pearsonr(product_stats[metric], product_stats['quality_score'])
            correlations[metric] = {
                'correlation': correlation[0],
                'p_value': correlation[1],
                'significant': correlation[1] < 0.05
            }
        
        # Identify strongest indicators
        significant_correlations = {
            k: v['correlation'] for k, v in correlations.items() 
            if v['significant'] and abs(v['correlation']) > 0.3
        }
        
        strongest_indicator = max(
            significant_correlations.items(),
            key=lambda x: abs(x[1])
        ) if significant_correlations else None
        
        return {
            'correlations': correlations,
            'significant_indicators': significant_correlations,
            'strongest_indicator': strongest_indicator,
            'quality_predictors': [
                k for k, v in correlations.items() 
                if v['significant'] and v['correlation'] > 0.5
            ]
        }
