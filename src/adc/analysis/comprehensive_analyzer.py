"""
Comprehensive Amazon Review Analysis Module

This module provides a unified interface for all analysis components:
1. Temporal Analysis (spikes, seasonality, trends)
2. User Behavior Analysis (superusers, influence, helpfulness)
3. Product Analysis (ingredients, organic trends, quality)
"""

from typing import List, Dict, Any
import json
from datetime import datetime

from ..models import AmazonReview
from .temporal_analyzer import TemporalAnalyzer
from .user_behavior_analyzer import UserBehaviorAnalyzer
from .product_analyzer import ProductAnalyzer


class ComprehensiveAnalyzer:
    """
    Main analysis class that coordinates all analytical components.
    """
    
    def __init__(self, reviews: List[AmazonReview]):
        """
        Initialize with review data.
        
        Args:
            reviews: List of AmazonReview objects
        """
        self.reviews = reviews
        self.temporal_analyzer = TemporalAnalyzer(reviews)
        self.user_analyzer = UserBehaviorAnalyzer(reviews)
        self.product_analyzer = ProductAnalyzer(reviews)
        
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive analysis across all components.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("🔄 Running Comprehensive Amazon Review Analysis...")
        print("=" * 50)
        
        analysis_results = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_reviews': len(self.reviews),
                'analysis_version': '1.0.0'
            }
        }
        
        # 1. Temporal Analysis
        print("📊 Running Temporal Analysis...")
        try:
            temporal_results = self._run_temporal_analysis()
            analysis_results['temporal_analysis'] = temporal_results
            print("✅ Temporal Analysis completed")
        except Exception as e:
            print(f"❌ Temporal Analysis failed: {str(e)}")
            analysis_results['temporal_analysis'] = {'error': str(e)}
        
        # 2. User Behavior Analysis
        print("👥 Running User Behavior Analysis...")
        try:
            user_results = self._run_user_analysis()
            analysis_results['user_behavior_analysis'] = user_results
            print("✅ User Behavior Analysis completed")
        except Exception as e:
            print(f"❌ User Behavior Analysis failed: {str(e)}")
            analysis_results['user_behavior_analysis'] = {'error': str(e)}
        
        # 3. Product Analysis
        print("🏷️ Running Product Analysis...")
        try:
            product_results = self._run_product_analysis()
            analysis_results['product_analysis'] = product_results
            print("✅ Product Analysis completed")
        except Exception as e:
            print(f"❌ Product Analysis failed: {str(e)}")
            analysis_results['product_analysis'] = {'error': str(e)}
        
        # 4. Generate Summary Insights
        print("💡 Generating Summary Insights...")
        try:
            insights = self._generate_insights(analysis_results)
            analysis_results['summary_insights'] = insights
            print("✅ Summary Insights generated")
        except Exception as e:
            print(f"❌ Insight generation failed: {str(e)}")
            analysis_results['summary_insights'] = {'error': str(e)}
        
        print("🎉 Comprehensive Analysis completed!")
        return analysis_results
    
    def _run_temporal_analysis(self) -> Dict[str, Any]:
        """Run all temporal analysis components."""
        return {
            'positive_review_spikes': self.temporal_analyzer.analyze_positive_review_spikes(),
            'seasonal_product_shifts': self.temporal_analyzer.analyze_seasonal_product_shifts(),
            'helpful_vote_increases': self.temporal_analyzer.analyze_helpful_vote_increases(),
            'keyword_frequency_changes': self.temporal_analyzer.analyze_keyword_frequency_changes()
        }
    
    def _run_user_analysis(self) -> Dict[str, Any]:
        """Run all user behavior analysis components."""
        return {
            'superusers_influencers': self.user_analyzer.identify_superusers_influencers(),
            'review_impact_on_sales': self.user_analyzer.analyze_review_impact_on_sales(),
            'helpful_review_characteristics': self.user_analyzer.analyze_helpful_review_characteristics(),
            'user_product_preferences': self.user_analyzer.analyze_user_product_preferences()
        }
    
    def _run_product_analysis(self) -> Dict[str, Any]:
        """Run all product analysis components."""
        return {
            'ingredient_correlation': self.product_analyzer.analyze_ingredient_correlation(),
            'organic_product_trends': self.product_analyzer.analyze_organic_product_trends(),
            'product_quality_indicators': self.product_analyzer.analyze_product_quality_indicators()
        }
    
    def _generate_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level insights from analysis results."""
        insights = {
            'key_findings': [],
            'recommendations': [],
            'alerts': [],
            'trends': {}
        }
        
        # Extract temporal insights
        temporal = results.get('temporal_analysis', {})
        if 'positive_review_spikes' in temporal:
            spikes = temporal['positive_review_spikes']
            if spikes.get('total_spike_days', 0) > 0:
                insights['key_findings'].append(
                    f"Detected {spikes['total_spike_days']} days with positive review spikes"
                )
                
                if spikes.get('recurring_patterns', {}).get('recurring_found'):
                    insights['trends']['recurring_spikes'] = True
                    insights['key_findings'].append("Found recurring positive review spike patterns")
        
        if 'keyword_frequency_changes' in temporal:
            keywords = temporal['keyword_frequency_changes']
            increasing_negative = keywords.get('increasing_negative_keywords', {})
            if increasing_negative:
                insights['alerts'].append(
                    f"Increasing negative keywords detected: {list(increasing_negative.keys())[:3]}"
                )
        
        # Extract user behavior insights
        user_behavior = results.get('user_behavior_analysis', {})
        if 'superusers_influencers' in user_behavior:
            superusers = user_behavior['superusers_influencers']
            if superusers.get('identified_superusers', 0) > 0:
                insights['key_findings'].append(
                    f"Identified {superusers['identified_superusers']} potential superusers/influencers"
                )
                
                comparison = superusers.get('superuser_comparison', {})
                if 'avg_helpful_votes' in comparison:
                    multiplier = comparison['avg_helpful_votes'].get('multiplier', 1)
                    if multiplier > 2:
                        insights['key_findings'].append(
                            f"Superusers receive {multiplier:.1f}x more helpful votes than average"
                        )
        
        if 'helpful_review_characteristics' in user_behavior:
            helpful_chars = user_behavior['helpful_review_characteristics']
            recommendations = helpful_chars.get('recommendations', [])
            insights['recommendations'].extend(recommendations)
        
        # Extract product insights
        product_analysis = results.get('product_analysis', {})
        if 'ingredient_correlation' in product_analysis:
            ingredients = product_analysis['ingredient_correlation']
            top_positive = ingredients.get('top_positive_ingredients', [])
            if top_positive:
                insights['key_findings'].append(
                    f"Top positive ingredients: {[ing[0] for ing in top_positive[:3]]}"
                )
            
            top_negative = ingredients.get('top_negative_ingredients', [])
            if top_negative:
                insights['alerts'].append(
                    f"Negatively correlated ingredients: {[ing[0] for ing in top_negative[:3]]}"
                )
        
        if 'organic_product_trends' in product_analysis:
            organic = product_analysis['organic_product_trends']
            trend_analysis = organic.get('trend_analysis', {})
            if 'organic_percentage_trend' in trend_analysis:
                trend = trend_analysis['organic_percentage_trend']
                if trend.get('direction') == 'increasing':
                    insights['trends']['organic_growth'] = True
                    insights['key_findings'].append("Organic product mentions are increasing over time")
        
        # Add recommendations
        insights['recommendations'].extend([
            "Monitor recurring spike patterns for marketing optimization",
            "Leverage superuser insights for influencer partnerships",
            "Focus on positively correlated ingredients in product development"
        ])
        
        return insights
    
    def export_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Export analysis results to JSON file.
        
        Args:
            results: Analysis results dictionary
            filename: Output filename (optional)
            
        Returns:
            Filename of exported results
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"amazon_review_analysis_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        json_results = self._prepare_for_json(results)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        return filename
    
    def _prepare_for_json(self, data: Any) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        elif hasattr(data, 'isoformat'):  # pandas Timestamp
            return data.isoformat()
        else:
            return data
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary report.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Formatted summary report as string
        """
        report = []
        report.append("=" * 60)
        report.append("AMAZON REVIEW ANALYSIS SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Date: {results['metadata']['analysis_timestamp']}")
        report.append(f"Total Reviews Analyzed: {results['metadata']['total_reviews']}")
        report.append("")
        
        # Key Findings
        insights = results.get('summary_insights', {})
        if 'key_findings' in insights:
            report.append("KEY FINDINGS:")
            report.append("-" * 15)
            for finding in insights['key_findings']:
                report.append(f"• {finding}")
            report.append("")
        
        # Alerts
        if 'alerts' in insights:
            report.append("ALERTS:")
            report.append("-" * 8)
            for alert in insights['alerts']:
                report.append(f"⚠️  {alert}")
            report.append("")
        
        # Recommendations
        if 'recommendations' in insights:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 17)
            for rec in insights['recommendations']:
                report.append(f"💡 {rec}")
            report.append("")
        
        # Temporal Analysis Summary
        temporal = results.get('temporal_analysis', {})
        if temporal:
            report.append("TEMPORAL ANALYSIS:")
            report.append("-" * 18)
            
            spikes = temporal.get('positive_review_spikes', {})
            if 'total_spike_days' in spikes:
                report.append(f"Positive Review Spikes: {spikes['total_spike_days']} days detected")
            
            seasonal = temporal.get('seasonal_product_shifts', {})
            if 'strongest_seasonal_trend' in seasonal and seasonal['strongest_seasonal_trend']:
                season, data = seasonal['strongest_seasonal_trend']
                report.append(f"Strongest Seasonal Trend: {season} (peak month: {data['peak_month']})")
            
            report.append("")
        
        # User Behavior Summary
        user_behavior = results.get('user_behavior_analysis', {})
        if user_behavior:
            report.append("USER BEHAVIOR ANALYSIS:")
            report.append("-" * 23)
            
            superusers = user_behavior.get('superusers_influencers', {})
            if 'identified_superusers' in superusers:
                report.append(f"Superusers Identified: {superusers['identified_superusers']}")
            
            report.append("")
        
        # Product Analysis Summary
        product = results.get('product_analysis', {})
        if product:
            report.append("PRODUCT ANALYSIS:")
            report.append("-" * 17)
            
            ingredients = product.get('ingredient_correlation', {})
            if 'top_positive_ingredients' in ingredients:
                top_ingredients = [ing[0] for ing in ingredients['top_positive_ingredients'][:3]]
                report.append(f"Top Positive Ingredients: {', '.join(top_ingredients)}")
            
            organic = product.get('organic_product_trends', {})
            overall_comp = organic.get('overall_comparison', {})
            if 'organic_percentage' in overall_comp:
                report.append(f"Organic Product Mentions: {overall_comp['organic_percentage']:.1f}%")
            
            report.append("")
        
        report.append("=" * 60)
        report.append("End of Report")
        report.append("=" * 60)
        
        return "\n".join(report)
