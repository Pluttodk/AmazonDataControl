#!/usr/bin/env python3
"""
Comprehensive Analysis Example for Amazon Review Data

This script demonstrates all analysis components:
1. Temporal Analysis - spikes, seasonality, keyword trends
2. User Behavior Analysis - superusers, influence, helpfulness patterns
3. Product Analysis - ingredient correlation, organic trends, quality indicators
"""

from src.adc.generation.score_generator import AmazonReviewGenerator
from src.adc.analysis.comprehensive_analyzer import ComprehensiveAnalyzer
from src.adc.analysis.visualizer import create_all_plots
import json


def generate_sample_data(n_reviews: int = 200) -> list:
    """Generate sample review data with realistic patterns."""
    print(f"🔄 Generating {n_reviews} sample Amazon reviews...")
    
    generator = AmazonReviewGenerator(seed=42)
    reviews = generator.generate_reviews(n_reviews=n_reviews)
    
    print(f"✅ Generated {len(reviews)} reviews with PyMC-based scores")
    return reviews


def run_temporal_analysis_demo(analyzer: ComprehensiveAnalyzer):
    """Demonstrate temporal analysis components."""
    print("\n" + "="*60)
    print("📊 TEMPORAL ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # 1. Positive Review Spikes
    print("\n🔍 Analyzing Positive Review Spikes...")
    spikes = analyzer.temporal_analyzer.analyze_positive_review_spikes()
    print(f"   📈 Detected {spikes['total_spike_days']} spike days")
    if spikes['recurring_patterns']['recurring_found']:
        print(f"   🔄 Recurring patterns found in months: {list(spikes['recurring_patterns']['recurring_months'].keys())}")
    
    # 2. Seasonal Product Shifts
    print("\n🌿 Analyzing Seasonal Product Shifts...")
    seasonal = analyzer.temporal_analyzer.analyze_seasonal_product_shifts()
    if seasonal['strongest_seasonal_trend']:
        season, data = seasonal['strongest_seasonal_trend']
        print(f"   🏆 Strongest seasonal trend: {season} (peak month: {data['peak_month']})")
    
    # 3. Helpful Vote Increases
    print("\n👍 Analyzing Helpful Vote Increases...")
    helpful_increases = analyzer.temporal_analyzer.analyze_helpful_vote_increases()
    trending_count = helpful_increases['products_with_increases']
    print(f"   📊 Found {trending_count} products with increasing helpful vote trends")
    
    # 4. Keyword Frequency Changes
    print("\n🔤 Analyzing Keyword Frequency Changes...")
    keyword_changes = analyzer.temporal_analyzer.analyze_keyword_frequency_changes()
    increasing_negative = keyword_changes['increasing_negative_keywords']
    if increasing_negative:
        print(f"   ⚠️  Increasing negative keywords: {list(increasing_negative.keys())[:3]}")
    else:
        print("   ✅ No concerning keyword trends detected")


def run_user_behavior_demo(analyzer: ComprehensiveAnalyzer):
    """Demonstrate user behavior analysis components."""
    print("\n" + "="*60)
    print("👥 USER BEHAVIOR ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # 1. Superusers/Influencers
    print("\n⭐ Identifying Superusers and Influencers...")
    superusers = analyzer.user_analyzer.identify_superusers_influencers()
    if 'identified_superusers' in superusers:
        print(f"   🎯 Identified {superusers['identified_superusers']} potential superusers")
        
        comparison = superusers.get('superuser_comparison', {})
        if 'avg_helpful_votes' in comparison:
            multiplier = comparison['avg_helpful_votes']['multiplier']
            print(f"   📊 Superusers get {multiplier:.1f}x more helpful votes than average")
    
    # 2. Review Impact on Sales
    print("\n💰 Analyzing Review Impact on Product Sales...")
    impact = analyzer.user_analyzer.analyze_review_impact_on_sales()
    if 'total_high_impact_reviews' in impact:
        print(f"   📈 Analyzed {impact['total_high_impact_reviews']} high-impact reviews")
        correlation = impact.get('correlation_helpful_votes_impact', {})
        if correlation and correlation['correlation'] > 0.3:
            print(f"   🔗 Strong correlation found: {correlation['correlation']:.3f}")
    
    # 3. Helpful Review Characteristics
    print("\n✨ Analyzing What Makes Reviews Helpful...")
    helpful_chars = analyzer.user_analyzer.analyze_helpful_review_characteristics()
    text_stats = helpful_chars['characteristics']['text_length']
    helpful_avg = text_stats['helpful_avg']
    non_helpful_avg = text_stats['non_helpful_avg']
    print(f"   📝 Helpful reviews avg length: {helpful_avg:.0f} chars")
    print(f"   📝 Non-helpful reviews avg length: {non_helpful_avg:.0f} chars")
    
    recommendations = helpful_chars.get('recommendations', [])
    if recommendations:
        print("   💡 Recommendations:")
        for rec in recommendations[:2]:
            print(f"      • {rec}")
    
    # 4. User Product Preferences
    print("\n🎯 Analyzing User Product Preferences...")
    preferences = analyzer.user_analyzer.analyze_user_product_preferences()
    total_preference_reviews = preferences['total_reviews_with_preferences']
    preference_rate = preferences['preference_rate'] * 100
    print(f"   🔍 Found {total_preference_reviews} reviews with specific preferences ({preference_rate:.1f}%)")
    
    top_features = preferences.get('top_requested_features', [])
    if top_features:
        print("   🏆 Top requested features:")
        for feature, data in top_features[:3]:
            print(f"      • {feature}: {data['total_mentions']} mentions")


def run_product_analysis_demo(analyzer: ComprehensiveAnalyzer):
    """Demonstrate product analysis components."""
    print("\n" + "="*60)
    print("🏷️ PRODUCT ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # 1. Ingredient Correlation
    print("\n🧪 Analyzing Ingredient Correlation with Reviews...")
    ingredients = analyzer.product_analyzer.analyze_ingredient_correlation()
    total_analyzed = ingredients['total_ingredients_analyzed']
    significant_corr = ingredients['significant_correlations']
    print(f"   🔬 Analyzed {total_analyzed} ingredients, found {significant_corr} significant correlations")
    
    top_positive = ingredients.get('top_positive_ingredients', [])
    if top_positive:
        print("   ✅ Top positive ingredients:")
        for ingredient, correlation in top_positive[:3]:
            print(f"      • {ingredient}: {correlation:.3f} correlation")
    
    top_negative = ingredients.get('top_negative_ingredients', [])
    if top_negative:
        print("   ❌ Negatively correlated ingredients:")
        for ingredient, correlation in top_negative[:2]:
            print(f"      • {ingredient}: {correlation:.3f} correlation")
    
    # 2. Organic Product Trends
    print("\n🌱 Analyzing Organic Product Trends...")
    organic = analyzer.product_analyzer.analyze_organic_product_trends()
    overall_comp = organic['overall_comparison']
    organic_percentage = overall_comp['organic_percentage']
    print(f"   🌿 Organic product mentions: {organic_percentage:.1f}% of all reviews")
    
    trend_analysis = organic.get('trend_analysis', {})
    if 'organic_percentage_trend' in trend_analysis:
        trend = trend_analysis['organic_percentage_trend']
        direction = trend['direction']
        change = trend['change']
        print(f"   📊 Trend: {direction} ({change:+.1f}% change over time)")
    
    # 3. Product Quality Indicators
    print("\n⭐ Analyzing Product Quality Indicators...")
    quality = analyzer.product_analyzer.analyze_product_quality_indicators()
    if 'total_products_analyzed' in quality:
        total_products = quality['total_products_analyzed']
        distribution = quality['quality_distribution']
        print(f"   📊 Analyzed {total_products} products")
        print(f"   🏆 High quality: {distribution['high_quality']} products")
        print(f"   ⚖️  Medium quality: {distribution['medium_quality']} products")
        print(f"   📉 Low quality: {distribution['low_quality']} products")
        
        indicators = quality['quality_indicators']
        strongest = indicators.get('strongest_indicator')
        if strongest:
            metric, correlation = strongest
            print(f"   🎯 Strongest quality indicator: {metric} ({correlation:.3f} correlation)")


def main():
    """Main demonstration function."""
    print("🚀 Amazon Review Comprehensive Analysis Demo")
    print("=" * 60)
    
    # Generate sample data
    reviews = generate_sample_data(n_reviews=1_000)
    
    # Initialize comprehensive analyzer
    analyzer = ComprehensiveAnalyzer(reviews)
    
    # Run individual analysis demonstrations
    run_temporal_analysis_demo(analyzer)
    run_user_behavior_demo(analyzer)
    run_product_analysis_demo(analyzer)
    
    # Run full comprehensive analysis
    print("\n" + "="*60)
    print("🎯 RUNNING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    results = analyzer.run_full_analysis()
    
    # Export results
    print("\n💾 Exporting Results...")
    results_file = analyzer.export_results(results)
    print(f"✅ Results exported to: {results_file}")
    
    # Create comprehensive visualizations
    print("\n🎨 Creating Comprehensive Visualizations...")
    visualizer = create_all_plots(results, output_dir="analysis_plots")
    print("✅ All stylistic plots have been saved to 'analysis_plots' directory")
    
    # Generate summary report
    print("\n📋 Generating Summary Report...")
    summary_report = analyzer.generate_summary_report(results)
    
    with open('analysis_summary_report.txt', 'w') as f:
        f.write(summary_report)
    
    print("✅ Summary report saved to: analysis_summary_report.txt")
    
    # Display key insights
    print("\n" + "="*60)
    print("💡 KEY INSIGHTS SUMMARY")
    print("="*60)
    
    insights = results.get('summary_insights', {})
    
    key_findings = insights.get('key_findings', [])
    if key_findings:
        print("\n🔍 Key Findings:")
        for finding in key_findings[:5]:
            print(f"   • {finding}")
    
    alerts = insights.get('alerts', [])
    if alerts:
        print("\n⚠️  Alerts:")
        for alert in alerts[:3]:
            print(f"   • {alert}")
    
    recommendations = insights.get('recommendations', [])
    if recommendations:
        print("\n💡 Recommendations:")
        for rec in recommendations[:3]:
            print(f"   • {rec}")
    
    print(f"\n🎉 Analysis completed! Check {results_file} for detailed results.")


if __name__ == "__main__":
    main()
