#!/usr/bin/env python3
"""
Example script demonstrating the PyMC-based Amazon review generation.

This script shows how to:
1. Generate deterministic scores using PyMC
2. Create complete Amazon review objects with placeholder data
3. Analyze the generated data distribution
"""

from src.adc.generation.score_generator import AmazonReviewGenerator, ScoreAnalyzer
import json


def main():
    """Main example function demonstrating review generation."""
    print("🔄 Amazon Review Generator Example")
    print("=" * 50)
    
    # Initialize the generator with a seed for reproducibility
    generator = AmazonReviewGenerator(seed=42)
    
    # Generate a batch of reviews
    print("📊 Generating 20 Amazon reviews with PyMC-based scores...")
    reviews = generator.generate_reviews(n_reviews=20)
    
    print(f"✅ Generated {len(reviews)} reviews successfully!")
    print()
    
    # Display first few reviews
    print("📋 Sample Generated Reviews:")
    print("-" * 30)
    for i, review in enumerate(reviews[:3]):
        print(f"\n🔸 Review {i+1}:")
        print(f"   Rating: {review.rating}/5 ⭐")
        print(f"   Title: {review.title}")
        print(f"   Helpful Votes: {review.helpful_vote}")
        print(f"   Verified Purchase: {review.verified_purchase}")
        print(f"   ASIN: {review.asin}")
        print(f"   Timestamp: {review.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Images: {len(review.images)} image(s)")
    
    # Analyze the generated data
    print("\n📈 Statistical Analysis:")
    print("-" * 25)
    analyzer = ScoreAnalyzer()
    stats = analyzer.analyze_score_distribution(reviews)
    
    print(f"Rating Statistics:")
    print(f"  • Mean Rating: {stats['rating_stats']['mean']:.2f}")
    print(f"  • Rating Std Dev: {stats['rating_stats']['std']:.2f}")
    print(f"  • Rating Distribution: {stats['rating_stats']['distribution']}")
    
    print(f"\nHelpful Vote Statistics:")
    print(f"  • Mean Helpful Votes: {stats['helpful_vote_stats']['mean']:.2f}")
    print(f"  • Helpful Votes Std Dev: {stats['helpful_vote_stats']['std']:.2f}")
    print(f"  • Max Helpful Votes: {stats['helpful_vote_stats']['max']}")
    print(f"  • Min Helpful Votes: {stats['helpful_vote_stats']['min']}")
    
    print(f"\n🔗 Rating-Helpful Vote Correlation: {stats['correlation']:.3f}")
    
    # Demonstrate deterministic generation
    print("\n🔄 Demonstrating Deterministic Generation:")
    print("-" * 40)
    print("Generating the same data twice with the same seed...")
    
    # First generation
    gen1 = AmazonReviewGenerator(seed=123)
    reviews1 = gen1.generate_reviews(n_reviews=5)
    
    # Second generation with same seed
    gen2 = AmazonReviewGenerator(seed=123)
    reviews2 = gen2.generate_reviews(n_reviews=5)
    
    # Compare ratings
    ratings1 = [r.rating for r in reviews1]
    ratings2 = [r.rating for r in reviews2]
    
    print(f"First generation ratings:  {ratings1}")
    print(f"Second generation ratings: {ratings2}")
    print(f"Ratings match: {ratings1 == ratings2} ✅" if ratings1 == ratings2 else "❌")
    
    # Export sample data to JSON
    print("\n💾 Exporting sample data...")
    sample_reviews = reviews[:5]
    export_data = []
    
    for review in sample_reviews:
        review_dict = review.model_dump()
        # Convert datetime to string for JSON serialization
        review_dict['timestamp'] = review_dict['timestamp'].isoformat()
        export_data.append(review_dict)
    
    with open('sample_reviews.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print("✅ Sample reviews exported to 'sample_reviews.json'")
    
    print("\n🎉 Example completed successfully!")


if __name__ == "__main__":
    main()
