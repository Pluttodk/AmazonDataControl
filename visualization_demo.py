#!/usr/bin/env python3
"""
Visualization Demo for Amazon Review Analysis

This script demonstrates the visualization capabilities by generating
sample data and creating all the stylistic matplotlib plots.
"""

from src.adc.generation.score_generator import AmazonReviewGenerator
from src.adc.analysis.comprehensive_analyzer import ComprehensiveAnalyzer
from src.adc.analysis.visualizer import create_all_plots
import json
import os


def main():
    """Generate data, run analysis, and create all visualizations."""
    print("🎨 Amazon Review Analysis Visualization Demo")
    print("=" * 60)
    
    # Generate sample data
    print("🔄 Generating sample Amazon reviews...")
    generator = AmazonReviewGenerator(seed=42)
    reviews = generator.generate_reviews(n_reviews=500)
    print(f"✅ Generated {len(reviews)} reviews")
    
    # Run comprehensive analysis
    print("\n🎯 Running comprehensive analysis...")
    analyzer = ComprehensiveAnalyzer(reviews)
    results = analyzer.run_full_analysis()
    print("✅ Analysis completed")
    
    # Create all visualizations
    print("\n🎨 Creating comprehensive visualizations...")
    output_dir = "visualization_output"
    visualizer = create_all_plots(results, output_dir=output_dir)
    
    # List created plots
    plot_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print(f"\n📊 Created {len(plot_files)} visualization plots:")
    for plot_file in sorted(plot_files):
        plot_name = plot_file.replace('.png', '').replace('_', ' ').title()
        print(f"   • {plot_name}")
    
    print(f"\n✅ All plots saved to '{output_dir}' directory")
    print("🎉 Visualization demo completed!")
    
    # Display some sample insights
    insights = results.get('summary_insights', {})
    findings = insights.get('key_findings', [])
    if findings:
        print(f"\n💡 Sample insights from the analysis:")
        for finding in findings[:3]:
            print(f"   • {finding}")


if __name__ == "__main__":
    main()
