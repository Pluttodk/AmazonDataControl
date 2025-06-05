"""
Visualization Module for Amazon Review Analysis

This module creates stylistic matplotlib plots based on analysis results:
1. Temporal plots - spikes, trends, seasonality
2. User behavior plots - superuser analysis, helpfulness patterns
3. Product plots - ingredient correlations, quality distributions
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import os


class AnalysisVisualizer:
    """Creates stylistic plots for Amazon review analysis results."""
    
    def __init__(self, output_dir: str = "analysis_plots"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        self._setup_style()
        self._create_output_dir()
    
    def _setup_style(self):
        """Set up matplotlib and seaborn styling for attractive plots."""
        # Set the style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Configure matplotlib settings
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def _create_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def plot_temporal_analysis(self, temporal_results: Dict[str, Any]):
        """Create temporal analysis plots."""
        print("📊 Creating temporal analysis plots...")
        
        # 1. Positive Review Spikes Plot
        self._plot_review_spikes(temporal_results.get('positive_review_spikes', {}))
        
        # 2. Seasonal Patterns Plot
        self._plot_seasonal_patterns(temporal_results.get('seasonal_product_shifts', {}))
        
        # 3. Helpful Vote Trends Plot
        self._plot_helpful_vote_trends(temporal_results.get('helpful_vote_increases', {}))
        
        # 4. Keyword Frequency Changes Plot
        self._plot_keyword_trends(temporal_results.get('keyword_frequency_changes', {}))
    
    def plot_user_behavior(self, user_results: Dict[str, Any]):
        """Create user behavior analysis plots."""
        print("👥 Creating user behavior plots...")
        
        # 1. Superuser Analysis Plot
        self._plot_superuser_analysis(user_results.get('superusers_influencers', {}))
        
        # 2. Review Helpfulness Characteristics
        self._plot_helpfulness_characteristics(user_results.get('helpful_review_characteristics', {}))
        
        # 3. User Preferences Plot
        self._plot_user_preferences(user_results.get('user_product_preferences', {}))
        
        # 4. Review Impact Analysis
        self._plot_review_impact(user_results.get('review_impact_on_sales', {}))
    
    def plot_product_analysis(self, product_results: Dict[str, Any]):
        """Create product analysis plots."""
        print("🏷️ Creating product analysis plots...")
        
        # 1. Ingredient Correlation Plot
        self._plot_ingredient_correlations(product_results.get('ingredient_correlation', {}))
        
        # 2. Organic Product Trends
        self._plot_organic_trends(product_results.get('organic_product_trends', {}))
        
        # 3. Product Quality Distribution
        self._plot_quality_distribution(product_results.get('product_quality_indicators', {}))
    
    def _plot_review_spikes(self, spikes_data: Dict[str, Any]):
        """Plot positive review spikes over time."""
        if not spikes_data or 'spike_details' not in spikes_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Monthly spike frequency
        monthly_spikes = spikes_data.get('monthly_spike_frequency', {})
        if monthly_spikes:
            months = list(monthly_spikes.keys())
            frequencies = list(monthly_spikes.values())
            
            ax1.bar(months, frequencies, color='coral', alpha=0.7, edgecolor='darkred')
            ax1.set_title('Positive Review Spikes by Month', fontweight='bold', pad=20)
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Number of Spike Days')
            ax1.set_xticks(range(1, 13))
            ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # Spike threshold visualization
        total_spikes = spikes_data.get('total_spike_days', 0)
        threshold = spikes_data.get('spike_threshold', 1.5)
        avg_spike_rate = spikes_data.get('avg_spike_positive_rate', 0)
        
        # Create a pie chart for spike vs normal days
        spike_data = [total_spikes, 365 - total_spikes]  # Assuming yearly data
        labels = [f'Spike Days\n({total_spikes})', f'Normal Days\n({365 - total_spikes})']
        colors = ['#ff6b6b', '#4ecdc4']
        
        ax2.pie(spike_data, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, textprops={'fontsize': 11})
        ax2.set_title(f'Review Spike Distribution (Threshold: {threshold}x)', 
                     fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/review_spikes_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_seasonal_patterns(self, seasonal_data: Dict[str, Any]):
        """Plot seasonal product patterns."""
        seasonal_patterns = seasonal_data.get('seasonal_patterns', {})
        if not seasonal_patterns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        for idx, (season, data) in enumerate(seasonal_patterns.items()):
            if idx >= 4:
                break
                
            monthly_dist = data.get('monthly_distribution', {})
            if monthly_dist:
                months = list(monthly_dist.keys())
                counts = list(monthly_dist.values())
                
                ax = axes[idx]
                bars = ax.bar(months, counts, color=colors[idx], alpha=0.7, 
                            edgecolor='black', linewidth=0.5)
                
                # Highlight peak month
                peak_month = data.get('peak_month', 0)
                if peak_month in months:
                    peak_idx = months.index(peak_month)
                    bars[peak_idx].set_color(colors[idx])
                    bars[peak_idx].set_alpha(1.0)
                    bars[peak_idx].set_edgecolor('gold')
                    bars[peak_idx].set_linewidth(3)
                
                ax.set_title(f'{season.title()} Product Mentions\n'
                           f'(Peak: Month {peak_month}, Rating: {data.get("avg_rating", 0):.1f})',
                           fontweight='bold')
                ax.set_xlabel('Month')
                ax.set_ylabel('Review Count')
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 
                                  'J', 'A', 'S', 'O', 'N', 'D'])
        
        # Hide unused subplots
        for idx in range(len(seasonal_patterns), 4):
            axes[idx].set_visible(False)
        
        plt.suptitle('Seasonal Product Review Patterns', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_helpful_vote_trends(self, helpful_data: Dict[str, Any]):
        """Plot helpful vote trend analysis."""
        trending_products = helpful_data.get('trending_products', {})
        if not trending_products:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top trending products
        top_products = sorted(trending_products.items(), 
                            key=lambda x: x[1]['trend_slope'], reverse=True)[:10]
        
        if top_products:
            product_names = [f"Product {i+1}" for i in range(len(top_products))]
            trend_slopes = [item[1]['trend_slope'] for item in top_products]
            improvements = [item[1]['improvement'] for item in top_products]
            
            # Trend slopes bar chart
            bars1 = ax1.barh(product_names, trend_slopes, color='lightblue', 
                           edgecolor='navy', alpha=0.7)
            ax1.set_title('Products with Increasing Helpful Vote Trends', fontweight='bold')
            ax1.set_xlabel('Trend Slope (Helpful Votes/Period)')
            ax1.grid(axis='x', alpha=0.3)
            
            # Color bars based on improvement magnitude
            for i, bar in enumerate(bars1):
                if trend_slopes[i] > np.mean(trend_slopes):
                    bar.set_color('lightgreen')
                    bar.set_alpha(0.8)
            
            # Improvement scatter plot
            ax2.scatter(trend_slopes, improvements, c=improvements, 
                       cmap='viridis', s=100, alpha=0.7, edgecolors='black')
            ax2.set_xlabel('Trend Slope')
            ax2.set_ylabel('Total Improvement (Helpful Votes)')
            ax2.set_title('Trend Slope vs. Total Improvement', fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(ax2.collections[0], ax=ax2)
            cbar.set_label('Improvement Magnitude')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/helpful_vote_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_keyword_trends(self, keyword_data: Dict[str, Any]):
        """Plot keyword frequency trends."""
        keyword_trends = keyword_data.get('keyword_trends', {})
        increasing_negative = keyword_data.get('increasing_negative_keywords', {})
        
        if not keyword_trends:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Overall keyword trend slopes
        keywords = list(keyword_trends.keys())
        slopes = [data['trend_slope'] for data in keyword_trends.values()]
        colors = ['red' if slope > 0 else 'green' for slope in slopes]
        
        bars = ax1.barh(keywords, slopes, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Keyword Frequency Trend Analysis', fontweight='bold', pad=20)
        ax1.set_xlabel('Trend Slope (Frequency Change Over Time)')
        
        # Highlight concerning keywords
        for i, (keyword, slope) in enumerate(zip(keywords, slopes)):
            if keyword in increasing_negative:
                bars[i].set_color('darkred')
                bars[i].set_alpha(0.9)
                bars[i].set_edgecolor('yellow')
                bars[i].set_linewidth(2)
        
        # Time series of most concerning keyword
        if increasing_negative:
            most_concerning = max(increasing_negative.items(), key=lambda x: x[1]['trend_slope'])
            keyword_name, keyword_info = most_concerning
            
            periods_data = keyword_info.get('periods', [])
            if periods_data:
                periods = [p['period'] for p in periods_data]
                frequencies = [p['frequency'] for p in periods_data]
                
                ax2.plot(periods, frequencies, marker='o', linewidth=3, 
                        markersize=8, color='darkred', label=f'"{keyword_name}"')
                ax2.fill_between(periods, frequencies, alpha=0.3, color='red')
                ax2.set_title(f'Most Concerning Keyword Trend: "{keyword_name}"', 
                            fontweight='bold', pad=20)
                ax2.set_xlabel('Time Period')
                ax2.set_ylabel('Frequency (Reviews Mentioning Keyword)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/keyword_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_superuser_analysis(self, superuser_data: Dict[str, Any]):
        """Plot superuser analysis results."""
        if not superuser_data or 'superuser_comparison' not in superuser_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Superuser vs regular user comparison
        comparison = superuser_data['superuser_comparison']
        metrics = list(comparison.keys())
        superuser_values = [comparison[metric]['superuser_avg'] for metric in metrics]
        population_values = [comparison[metric]['population_avg'] for metric in metrics]
        multipliers = [comparison[metric]['multiplier'] for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, superuser_values, width, label='Superusers', 
                       color='gold', alpha=0.8, edgecolor='darkorange')
        bars2 = ax1.bar(x + width/2, population_values, width, label='Average Users', 
                       color='lightblue', alpha=0.8, edgecolor='navy')
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Average Values')
        ax1.set_title('Superusers vs. Average Users Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Multiplier effect visualization
        bars3 = ax2.bar(metrics, multipliers, color='coral', alpha=0.7, 
                       edgecolor='darkred', linewidth=2)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline (1x)')
        ax2.set_ylabel('Multiplier (Superuser / Average)')
        ax2.set_title('Superuser Advantage Multipliers', fontweight='bold')
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Highlight bars above 2x multiplier
        for i, (bar, mult) in enumerate(zip(bars3, multipliers)):
            if mult > 2:
                bar.set_color('red')
                bar.set_alpha(0.9)
                # Add text annotation
                ax2.annotate(f'{mult:.1f}x', (i, mult + 0.1), 
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/superuser_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_helpfulness_characteristics(self, helpful_data: Dict[str, Any]):
        """Plot review helpfulness characteristics."""
        characteristics = helpful_data.get('characteristics', {})
        if not characteristics:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Text length comparison
        text_length = characteristics.get('text_length', {})
        if text_length:
            categories = ['Helpful Reviews', 'Non-Helpful Reviews']
            avg_lengths = [text_length['helpful_avg'], text_length['non_helpful_avg']]
            median_lengths = [text_length['helpful_median'], text_length['non_helpful_median']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax1.bar(x - width/2, avg_lengths, width, label='Average', 
                   color='lightgreen', alpha=0.7, edgecolor='darkgreen')
            ax1.bar(x + width/2, median_lengths, width, label='Median', 
                   color='lightcoral', alpha=0.7, edgecolor='darkred')
            
            ax1.set_ylabel('Text Length (Characters)')
            ax1.set_title('Text Length: Helpful vs Non-Helpful Reviews', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
        
        # 2. Rating distribution for helpful reviews
        rating_dist = characteristics.get('rating_distribution', {})
        if rating_dist:
            helpful_dist = rating_dist.get('helpful_reviews', {})
            non_helpful_dist = rating_dist.get('non_helpful_reviews', {})
            
            ratings = sorted(set(list(helpful_dist.keys()) + list(non_helpful_dist.keys())))
            helpful_counts = [helpful_dist.get(r, 0) for r in ratings]
            non_helpful_counts = [non_helpful_dist.get(r, 0) for r in ratings]
            
            x = np.arange(len(ratings))
            width = 0.35
            
            ax2.bar(x - width/2, helpful_counts, width, label='Helpful', 
                   color='gold', alpha=0.8, edgecolor='orange')
            ax2.bar(x + width/2, non_helpful_counts, width, label='Non-Helpful', 
                   color='lightblue', alpha=0.8, edgecolor='blue')
            
            ax2.set_xlabel('Rating')
            ax2.set_ylabel('Count')
            ax2.set_title('Rating Distribution by Helpfulness', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{r}★' for r in ratings])
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. Verified purchase impact
        verified_impact = characteristics.get('verified_purchase_impact', {})
        if verified_impact:
            categories = ['Helpful Reviews', 'Non-Helpful Reviews']
            verified_rates = [verified_impact['helpful_verified_rate'] * 100, 
                            verified_impact['non_helpful_verified_rate'] * 100]
            
            bars = ax3.bar(categories, verified_rates, color=['green', 'red'], 
                          alpha=0.7, edgecolor='black', linewidth=2)
            ax3.set_ylabel('Verified Purchase Rate (%)')
            ax3.set_title('Verified Purchase Impact on Helpfulness', fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            
            # Add percentage labels on bars
            for bar, rate in zip(bars, verified_rates):
                height = bar.get_height()
                ax3.annotate(f'{rate:.1f}%', (bar.get_x() + bar.get_width()/2, height),
                           ha='center', va='bottom', fontweight='bold')
        
        # 4. Rating extremes analysis
        rating_extremes = characteristics.get('rating_extremes', {})
        if rating_extremes:
            categories = ['Extreme Ratings\n(1★ or 5★)', 'Moderate Ratings\n(2★, 3★, 4★)']
            helpful_rates = [rating_extremes['extreme_helpful_rate'] * 100,
                           rating_extremes['moderate_helpful_rate'] * 100]
            
            bars = ax4.bar(categories, helpful_rates, color=['purple', 'orange'], 
                          alpha=0.7, edgecolor='black', linewidth=2)
            ax4.set_ylabel('Helpfulness Rate (%)')
            ax4.set_title('Rating Extremes vs. Helpfulness', fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            
            # Add percentage labels
            for bar, rate in zip(bars, helpful_rates):
                height = bar.get_height()
                ax4.annotate(f'{rate:.1f}%', (bar.get_x() + bar.get_width()/2, height),
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/helpfulness_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_user_preferences(self, preferences_data: Dict[str, Any]):
        """Plot user product preferences analysis."""
        keyword_analysis = preferences_data.get('keyword_analysis', {})
        top_features = preferences_data.get('top_requested_features', [])
        
        if not keyword_analysis and not top_features:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Top requested features
        if top_features:
            features = [item[0] for item in top_features[:10]]
            mentions = [item[1]['total_mentions'] for item in top_features[:10]]
            avg_ratings = [item[1]['avg_rating'] for item in top_features[:10]]
            
            # Create color map based on ratings
            colors = plt.cm.RdYlGn([r/5 for r in avg_ratings])
            
            bars = ax1.barh(features, mentions, color=colors, alpha=0.8, edgecolor='black')
            ax1.set_xlabel('Number of Mentions')
            ax1.set_title('Top Requested Product Features', fontweight='bold', pad=20)
            ax1.grid(axis='x', alpha=0.3)
            
            # Add rating annotations
            for i, (bar, rating) in enumerate(zip(bars, avg_ratings)):
                width = bar.get_width()
                ax1.annotate(f'{rating:.1f}★', (width + max(mentions)*0.01, bar.get_y() + bar.get_height()/2),
                           ha='left', va='center', fontweight='bold', fontsize=9)
        
        # Preference mentions over time (if trend data available)
        if keyword_analysis:
            # Select top 5 preferences with trend data
            trending_prefs = []
            for keyword, data in keyword_analysis.items():
                trend_data = data.get('trend_over_time', {})
                if trend_data and 'monthly_counts' in trend_data:
                    trending_prefs.append((keyword, data, trend_data))
            
            if trending_prefs:
                trending_prefs = trending_prefs[:5]  # Top 5
                
                for keyword, data, trend_data in trending_prefs:
                    monthly_counts = trend_data['monthly_counts']
                    if monthly_counts:
                        months = list(monthly_counts.keys())
                        counts = list(monthly_counts.values())
                        
                        # Convert period strings to numeric for plotting
                        month_nums = range(len(months))
                        ax2.plot(month_nums, counts, marker='o', linewidth=2, 
                               label=keyword.replace('_', ' ').title(), alpha=0.8)
                
                ax2.set_xlabel('Time Period')
                ax2.set_ylabel('Mentions Count')
                ax2.set_title('Preference Keywords Trend Over Time', fontweight='bold', pad=20)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/user_preferences.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_review_impact(self, impact_data: Dict[str, Any]):
        """Plot review impact on sales analysis."""
        if not impact_data or 'top_impactful_reviews' not in impact_data:
            return
        
        top_impacts = impact_data['top_impactful_reviews']
        correlation_data = impact_data.get('correlation_helpful_votes_impact', {})
        
        if not top_impacts:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top impactful reviews
        review_ids = [f"Review {i+1}" for i in range(len(top_impacts[:10]))]
        velocity_increases = [item['velocity_increase'] for item in top_impacts[:10]]
        helpful_votes = [item['helpful_votes'] for item in top_impacts[:10]]
        
        # Scatter plot: helpful votes vs velocity increase
        scatter = ax1.scatter(helpful_votes, velocity_increases, 
                            c=velocity_increases, cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Helpful Votes')
        ax1.set_ylabel('Velocity Increase (Reviews/Day)')
        ax1.set_title('Review Impact: Helpful Votes vs. Sales Velocity', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Velocity Increase')
        
        # Impact distribution
        impact_dist = impact_data.get('impact_distribution', {})
        if impact_dist:
            categories = ['Positive\nImpact', 'Negative\nImpact', 'No\nImpact']
            counts = [impact_dist['positive_impacts'], 
                     impact_dist['negative_impacts'], 
                     impact_dist['no_impact']]
            colors = ['green', 'red', 'gray']
            
            wedges, texts, autotexts = ax2.pie(counts, labels=categories, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            ax2.set_title('Distribution of Review Impact Types', fontweight='bold')
            
            # Style the text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/review_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ingredient_correlations(self, ingredient_data: Dict[str, Any]):
        """Plot ingredient correlation analysis."""
        ingredient_analysis = ingredient_data.get('ingredient_analysis', {})
        top_positive = ingredient_data.get('top_positive_ingredients', [])
        top_negative = ingredient_data.get('top_negative_ingredients', [])
        
        if not ingredient_analysis:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Top positive and negative correlations
        all_ingredients = []
        all_correlations = []
        
        for ingredient, correlation in top_positive[:8]:
            all_ingredients.append(ingredient)
            all_correlations.append(correlation)
        
        for ingredient, correlation in top_negative[:8]:
            all_ingredients.append(ingredient)
            all_correlations.append(correlation)
        
        if all_ingredients:
            colors = ['green' if corr > 0 else 'red' for corr in all_correlations]
            bars = ax1.barh(all_ingredients, all_correlations, color=colors, 
                           alpha=0.7, edgecolor='black')
            
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax1.set_xlabel('Correlation with Positive Reviews')
            ax1.set_title('Ingredient Correlation Analysis', fontweight='bold', pad=20)
            ax1.grid(axis='x', alpha=0.3)
            
            # Add correlation values as annotations
            for i, (bar, corr) in enumerate(zip(bars, all_correlations)):
                width = bar.get_width()
                x_pos = width + (0.01 if width >= 0 else -0.01)
                ax1.annotate(f'{corr:.3f}', (x_pos, bar.get_y() + bar.get_height()/2),
                           ha='left' if width >= 0 else 'right', va='center', 
                           fontweight='bold', fontsize=9)
        
        # Effect size analysis
        effect_sizes = []
        ingredient_names = []
        
        for ingredient, data in ingredient_analysis.items():
            effect_size = data.get('effect_size', {})
            rating_diff = effect_size.get('rating_difference', 0)
            if abs(rating_diff) > 0.01:  # Filter small effects
                effect_sizes.append(rating_diff)
                ingredient_names.append(ingredient)
        
        if effect_sizes:
            # Sort by absolute effect size
            sorted_data = sorted(zip(ingredient_names, effect_sizes), 
                               key=lambda x: abs(x[1]), reverse=True)
            ingredient_names, effect_sizes = zip(*sorted_data[:10])
            
            colors = ['darkgreen' if eff > 0 else 'darkred' for eff in effect_sizes]
            bars = ax2.barh(ingredient_names, effect_sizes, color=colors, 
                           alpha=0.8, edgecolor='black')
            
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Rating Difference (vs. Non-Ingredient Reviews)')
            ax2.set_title('Ingredient Effect on Average Rating', fontweight='bold', pad=20)
            ax2.grid(axis='x', alpha=0.3)
            
            # Add effect size annotations
            for bar, eff in zip(bars, effect_sizes):
                width = bar.get_width()
                x_pos = width + (0.01 if width >= 0 else -0.01)
                ax2.annotate(f'{eff:+.2f}', (x_pos, bar.get_y() + bar.get_height()/2),
                           ha='left' if width >= 0 else 'right', va='center', 
                           fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ingredient_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_organic_trends(self, organic_data: Dict[str, Any]):
        """Plot organic product trends."""
        trend_analysis = organic_data.get('trend_analysis', {})
        overall_comparison = organic_data.get('overall_comparison', {})
        period_analysis = organic_data.get('period_analysis', {})
        
        if not any([trend_analysis, overall_comparison, period_analysis]):
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Organic vs non-organic comparison
        if overall_comparison:
            categories = ['Organic Products', 'Non-Organic Products']
            counts = [overall_comparison['organic_count'], overall_comparison['non_organic_count']]
            colors = ['lightgreen', 'lightcoral']
            
            wedges, texts, autotexts = ax1.pie(counts, labels=categories, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            ax1.set_title('Organic vs Non-Organic Review Distribution', fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_fontweight('bold')
        
        # Rating comparison
        if overall_comparison:
            rating_comp = overall_comparison.get('rating_comparison', {})
            if rating_comp:
                categories = ['Organic', 'Non-Organic']
                ratings = [rating_comp['organic_avg'], rating_comp['non_organic_avg']]
                
                bars = ax2.bar(categories, ratings, color=['green', 'red'], 
                              alpha=0.7, edgecolor='black', linewidth=2)
                ax2.set_ylabel('Average Rating')
                ax2.set_title('Rating Comparison: Organic vs Non-Organic', fontweight='bold')
                ax2.set_ylim(0, 5)
                ax2.grid(axis='y', alpha=0.3)
                
                # Add rating labels
                for bar, rating in zip(bars, ratings):
                    height = bar.get_height()
                    ax2.annotate(f'{rating:.2f}★', (bar.get_x() + bar.get_width()/2, height),
                               ha='center', va='bottom', fontweight='bold')
        
        # Trend over time
        if period_analysis:
            periods = []
            organic_percentages = []
            
            for period_name, period_data in period_analysis.items():
                periods.append(period_name.replace('period_', 'P'))
                stats = period_data.get('stats', {})
                organic_percentages.append(stats.get('organic_percentage', 0))
            
            if periods and organic_percentages:
                ax3.plot(periods, organic_percentages, marker='o', linewidth=3, 
                        markersize=8, color='green', alpha=0.8)
                ax3.fill_between(periods, organic_percentages, alpha=0.3, color='green')
                ax3.set_xlabel('Time Period')
                ax3.set_ylabel('Organic Mentions (%)')
                ax3.set_title('Organic Product Mentions Trend', fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                # Add trend line
                if len(organic_percentages) > 1:
                    z = np.polyfit(range(len(organic_percentages)), organic_percentages, 1)
                    p = np.poly1d(z)
                    ax3.plot(periods, p(range(len(periods))), "--", color='darkgreen', 
                           alpha=0.8, linewidth=2, label=f'Trend: {z[0]:+.1f}%/period')
                    ax3.legend()
        
        # Keyword frequency analysis
        keyword_freq = organic_data.get('keyword_frequency', {})
        if keyword_freq:
            # Top 10 organic keywords
            sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            keywords, frequencies = zip(*sorted_keywords)
            
            bars = ax4.barh(keywords, frequencies, color='forestgreen', 
                           alpha=0.7, edgecolor='darkgreen')
            ax4.set_xlabel('Frequency')
            ax4.set_title('Top Organic/Natural Keywords', fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
            
            # Add frequency labels
            for bar, freq in zip(bars, frequencies):
                width = bar.get_width()
                ax4.annotate(f'{freq}', (width + max(frequencies)*0.01, 
                           bar.get_y() + bar.get_height()/2),
                           ha='left', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/organic_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_distribution(self, quality_data: Dict[str, Any]):
        """Plot product quality distribution analysis."""
        quality_distribution = quality_data.get('quality_distribution', {})
        quality_patterns = quality_data.get('quality_patterns', {})
        quality_indicators = quality_data.get('quality_indicators', {})
        
        if not any([quality_distribution, quality_patterns, quality_indicators]):
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Quality distribution pie chart
        if quality_distribution:
            categories = ['High Quality', 'Medium Quality', 'Low Quality']
            counts = [quality_distribution['high_quality'], 
                     quality_distribution['medium_quality'], 
                     quality_distribution['low_quality']]
            colors = ['gold', 'silver', 'lightcoral']
            
            wedges, texts, autotexts = ax1.pie(counts, labels=categories, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            ax1.set_title('Product Quality Distribution', fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_fontweight('bold')
        
        # Quality patterns comparison
        if quality_patterns:
            quality_levels = ['High Quality', 'Medium Quality', 'Low Quality']
            avg_ratings = []
            avg_helpful_votes = []
            
            for level in ['high_quality', 'medium_quality', 'low_quality']:
                pattern_data = quality_patterns.get(level, {})
                avg_ratings.append(pattern_data.get('avg_rating', 0))
                avg_helpful_votes.append(pattern_data.get('avg_helpful_votes', 0))
            
            x = np.arange(len(quality_levels))
            width = 0.35
            
            ax2.bar(x - width/2, avg_ratings, width, label='Avg Rating', 
                   color='lightblue', alpha=0.8, edgecolor='navy')
            
            # Secondary y-axis for helpful votes
            ax2_twin = ax2.twinx()
            ax2_twin.bar(x + width/2, avg_helpful_votes, width, label='Avg Helpful Votes', 
                        color='lightcoral', alpha=0.8, edgecolor='darkred')
            
            ax2.set_xlabel('Quality Level')
            ax2.set_ylabel('Average Rating', color='navy')
            ax2_twin.set_ylabel('Average Helpful Votes', color='darkred')
            ax2.set_title('Quality Patterns Comparison', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(quality_levels)
            
            # Add legends
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
        
        # Quality indicators correlation
        if quality_indicators:
            correlations = quality_indicators.get('correlations', {})
            if correlations:
                metrics = list(correlations.keys())
                corr_values = [correlations[metric]['correlation'] for metric in metrics]
                significance = [correlations[metric]['significant'] for metric in metrics]
                
                colors = ['green' if sig else 'red' for sig in significance]
                bars = ax3.barh(metrics, corr_values, color=colors, alpha=0.7, edgecolor='black')
                
                ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax3.set_xlabel('Correlation with Quality Score')
                ax3.set_title('Quality Indicators Correlation Analysis', fontweight='bold')
                ax3.grid(axis='x', alpha=0.3)
                
                # Add correlation values
                for bar, corr, sig in zip(bars, corr_values, significance):
                    width = bar.get_width()
                    x_pos = width + (0.02 if width >= 0 else -0.02)
                    symbol = '✓' if sig else '✗'
                    ax3.annotate(f'{corr:.3f} {symbol}', 
                               (x_pos, bar.get_y() + bar.get_height()/2),
                               ha='left' if width >= 0 else 'right', va='center', 
                               fontweight='bold', fontsize=9)
        
        # Top and bottom quality products comparison
        if quality_patterns:
            high_quality = quality_patterns.get('high_quality', {})
            low_quality = quality_patterns.get('low_quality', {})
            
            if high_quality and low_quality:
                categories = ['Avg Rating', 'Avg Helpful Votes', 'Verified Rate']
                high_values = [high_quality.get('avg_rating', 0),
                             high_quality.get('avg_helpful_votes', 0),
                             high_quality.get('avg_verified_rate', 0)]
                low_values = [low_quality.get('avg_rating', 0),
                            low_quality.get('avg_helpful_votes', 0),
                            low_quality.get('avg_verified_rate', 0)]
                
                x = np.arange(len(categories))
                width = 0.35
                
                ax4.bar(x - width/2, high_values, width, label='High Quality', 
                       color='gold', alpha=0.8, edgecolor='orange')
                ax4.bar(x + width/2, low_values, width, label='Low Quality', 
                       color='lightcoral', alpha=0.8, edgecolor='red')
                
                ax4.set_ylabel('Values')
                ax4.set_title('High vs Low Quality Products', fontweight='bold')
                ax4.set_xticks(x)
                ax4.set_xticklabels(categories)
                ax4.legend()
                ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/quality_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self, results: Dict[str, Any]):
        """Create a summary dashboard with key metrics."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create a 3x4 grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Extract key metrics
        metadata = results.get('metadata', {})
        insights = results.get('summary_insights', {})
        temporal = results.get('temporal_analysis', {})
        user_behavior = results.get('user_behavior_analysis', {})
        product = results.get('product_analysis', {})
        
        # Title
        fig.suptitle('Amazon Review Analysis Dashboard', fontsize=24, fontweight='bold', y=0.95)
        
        # Summary metrics (top row)
        ax1 = fig.add_subplot(gs[0, 0])
        total_reviews = metadata.get('total_reviews', 0)
        ax1.text(0.5, 0.5, f'{total_reviews:,}\nTotal Reviews', ha='center', va='center', 
                fontsize=20, fontweight='bold', transform=ax1.transAxes)
        ax1.set_title('Dataset Size', fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        spikes = temporal.get('positive_review_spikes', {})
        spike_days = spikes.get('total_spike_days', 0)
        ax2.text(0.5, 0.5, f'{spike_days}\nSpike Days', ha='center', va='center', 
                fontsize=20, fontweight='bold', transform=ax2.transAxes, color='red')
        ax2.set_title('Review Spikes', fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        superusers = user_behavior.get('superusers_influencers', {})
        super_count = superusers.get('identified_superusers', 0)
        ax3.text(0.5, 0.5, f'{super_count}\nSuperusers', ha='center', va='center', 
                fontsize=20, fontweight='bold', transform=ax3.transAxes, color='gold')
        ax3.set_title('Influencers Found', fontweight='bold')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        organic = product.get('organic_product_trends', {})
        organic_pct = organic.get('overall_comparison', {}).get('organic_percentage', 0)
        ax4.text(0.5, 0.5, f'{organic_pct:.1f}%\nOrganic', ha='center', va='center', 
                fontsize=20, fontweight='bold', transform=ax4.transAxes, color='green')
        ax4.set_title('Organic Products', fontweight='bold')
        ax4.axis('off')
        
        # Key findings (middle section)
        ax5 = fig.add_subplot(gs[1, :2])
        key_findings = insights.get('key_findings', [])[:5]
        if key_findings:
            findings_text = '\n'.join([f'• {finding}' for finding in key_findings])
            ax5.text(0.05, 0.95, findings_text, ha='left', va='top', 
                    fontsize=11, transform=ax5.transAxes, wrap=True)
        ax5.set_title('Key Findings', fontweight='bold', fontsize=14)
        ax5.axis('off')
        
        # Alerts (middle section)
        ax6 = fig.add_subplot(gs[1, 2:])
        alerts = insights.get('alerts', [])[:3]
        if alerts:
            alerts_text = '\n'.join([f'⚠️ {alert}' for alert in alerts])
            ax6.text(0.05, 0.95, alerts_text, ha='left', va='top', 
                    fontsize=11, transform=ax6.transAxes, wrap=True, color='red')
        ax6.set_title('Alerts', fontweight='bold', fontsize=14)
        ax6.axis('off')
        
        # Bottom row - mini visualizations
        # Rating distribution
        ax7 = fig.add_subplot(gs[2, 0])
        # Placeholder for rating distribution
        ratings = [1, 2, 3, 4, 5]
        counts = [10, 15, 25, 35, 15]  # Example data
        ax7.bar(ratings, counts, color='skyblue', alpha=0.7, edgecolor='navy')
        ax7.set_title('Rating Distribution', fontweight='bold')
        ax7.set_xlabel('Rating')
        ax7.set_ylabel('Count')
        
        # Temporal trend
        ax8 = fig.add_subplot(gs[2, 1])
        # Placeholder for temporal trend
        months = range(1, 13)
        trend = [20 + 5*np.sin(m/2) + np.random.normal(0, 2) for m in months]
        ax8.plot(months, trend, marker='o', color='green', linewidth=2)
        ax8.set_title('Monthly Review Trend', fontweight='bold')
        ax8.set_xlabel('Month')
        ax8.set_ylabel('Reviews')
        
        # Quality distribution
        ax9 = fig.add_subplot(gs[2, 2])
        quality_dist = product.get('product_quality_indicators', {}).get('quality_distribution', {})
        if quality_dist:
            labels = ['High', 'Medium', 'Low']
            sizes = [quality_dist.get('high_quality', 0), 
                    quality_dist.get('medium_quality', 0), 
                    quality_dist.get('low_quality', 0)]
            colors = ['gold', 'silver', 'lightcoral']
            ax9.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%')
        ax9.set_title('Quality Distribution', fontweight='bold')
        
        # Recommendations
        ax10 = fig.add_subplot(gs[2, 3])
        recommendations = insights.get('recommendations', [])[:3]
        if recommendations:
            rec_text = '\n'.join([f'💡 {rec[:40]}...' if len(rec) > 40 else f'💡 {rec}' 
                                for rec in recommendations])
            ax10.text(0.05, 0.95, rec_text, ha='left', va='top', 
                     fontsize=9, transform=ax10.transAxes, wrap=True)
        ax10.set_title('Recommendations', fontweight='bold')
        ax10.axis('off')
        
        plt.savefig(f'{self.output_dir}/summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Summary dashboard saved to {self.output_dir}/summary_dashboard.png")


def create_all_plots(results: Dict[str, Any], output_dir: str = "analysis_plots"):
    """
    Create all analysis plots from comprehensive results.
    
    Args:
        results: Complete analysis results dictionary
        output_dir: Directory to save plots
    """
    visualizer = AnalysisVisualizer(output_dir)
    
    print("🎨 Creating comprehensive analysis visualizations...")
    
    # Create temporal plots
    temporal_results = results.get('temporal_analysis', {})
    if temporal_results:
        visualizer.plot_temporal_analysis(temporal_results)
    
    # Create user behavior plots
    user_results = results.get('user_behavior_analysis', {})
    if user_results:
        visualizer.plot_user_behavior(user_results)
    
    # Create product analysis plots
    product_results = results.get('product_analysis', {})
    if product_results:
        visualizer.plot_product_analysis(product_results)
    
    # Create summary dashboard
    visualizer.create_summary_dashboard(results)
    
    print(f"✅ All plots saved to '{output_dir}' directory")
    return visualizer
