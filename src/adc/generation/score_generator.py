import pymc as pm
import numpy as np
import pytensor.tensor as pt
from datetime import datetime, timedelta
import random, uuid
import string
from typing import List, Dict, Any
from ..models import AmazonReview


class AmazonReviewGenerator:
    """
    PyMC-based generator for Amazon review data with deterministic score and helpful vote generation.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the review generator with a random seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible generation
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_deterministic_scores(self, n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate deterministic rating scores and helpful votes using PyMC.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary containing 'ratings' and 'helpful_votes' arrays
        """
        with pm.Model() as model:
            # Beta used for 0 to 1 scale
            product_quality = pm.Beta('product_quality', alpha=2, beta=1)
            
            # Rating Rough estimates
            rating_probs = pm.math.stack([
                0.05 * (1 - product_quality), 
                0.10 * (1 - product_quality),  
                0.15,                          
                0.30 + 0.20 * product_quality, 
                0.40 + 0.50 * product_quality 
            ])
            rating_probs = rating_probs / pm.math.sum(rating_probs)
            
            ratings = pm.Categorical('ratings', p=rating_probs, shape=n_samples)
            
            # Helpful votes - correlated with rating quality and review age
            # Higher ratings tend to get more helpful votes
            rating_effect = pm.Deterministic('rating_effect', ratings / 5.0)
            
            # Base helpful vote rate
            base_helpful_rate = pm.Exponential('base_helpful_rate', lam=0.5)
            
            # Helpful votes follow negative binomial (overdispersed Poisson)
            helpful_rate = base_helpful_rate * (1 + rating_effect)
            helpful_votes = pm.NegativeBinomial('helpful_votes', 
                                              mu=helpful_rate, 
                                              alpha=2, 
                                              shape=n_samples)
            
            # Sample from the model (We use shape rather than size to make the independent samples)
            trace = pm.sample(1, random_seed=self.seed, chains=1, 
                            tune=100, progressbar=False, return_inferencedata=False)
        
        # Convert to 1-5 scale for ratings (PyMC Categorical is 0-indexed)
        ratings_samples = trace['ratings'][0] + 1
        helpful_votes_samples = trace['helpful_votes'][0]
        
        return {
            'ratings': ratings_samples,
            'helpful_votes': helpful_votes_samples,
            'product_quality': trace['product_quality'][0]
        }
    
    def generate_placeholder_data(self) -> Dict[str, Any]:
        """
        Generate placeholder data for other review fields.
        
        Returns:
            Dictionary with placeholder values for non-PyMC fields
        """
        # Sample review titles
        titles = [
            "Great product!", "Love it!", "Excellent quality", "Highly recommend",
            "Good value for money", "Perfect!", "Amazing", "Not what I expected",
            "Could be better", "Fantastic purchase", "Decent product", "Outstanding!"
        ]
        
        # Sample review texts  
        texts = [
            "This product exceeded my expectations. Great quality and fast shipping.",
            "Exactly what I was looking for. Works perfectly.",
            "Good product overall, though delivery took a while.",
            "Amazing quality for the price. Would buy again.",
            "Product is okay but not amazing. Does the job.",
            "Fantastic! This is exactly what I needed.",
            "Great value for money. Highly recommend to others.",
            "Product quality is excellent. Very satisfied with purchase.",
        ]
        
        return {
            'title': random.choice(titles),
            'text': random.choice(texts),
            'images': self._generate_image_urls(),
            'asin': self._generate_asin(),
            'parent_asin': self._generate_asin(), 
            'user_id': self._generate_user_id(),
            'timestamp': self._generate_timestamp(),
            'verified_purchase': random.choice([True, False])
        }
    
    def _generate_asin(self) -> str:
        """Generate a realistic ASIN (10 character alphanumeric)."""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    
    def _generate_user_id(self) -> str:
        """Generate a realistic user ID."""
        return f"USER{uuid.uuid4().hex[:10].upper()}"
    
    def _generate_timestamp(self) -> datetime:
        """Generate a random timestamp within the last 2 years."""
        start_date = datetime.now() - timedelta(days=730)
        random_days = random.randint(0, 730)
        return start_date + timedelta(days=random_days)
    
    def _generate_image_urls(self) -> List[str]:
        """Generate random image URLs (0-3 images per review)."""
        num_images = random.randint(0, 3)
        return [f"temp{uuid.uuid4().hex[:8]}.jpg" for _ in range(num_images)]

    def generate_reviews(self, n_reviews: int = 10) -> List[AmazonReview]:
        """
        Generate a list of complete Amazon reviews with PyMC-based scores.
        
        Args:
            n_reviews: Number of reviews to generate
            
        Returns:
            List of AmazonReview objects
        """
        # Generate PyMC-based scores
        score_data = self.generate_deterministic_scores(n_reviews)
        
        reviews = []
        for i in range(n_reviews):
            # Get placeholder data
            placeholder_data = self.generate_placeholder_data()
            
            # Combine with PyMC-generated scores
            review_data = {
                'rating': int(score_data['ratings'][i]),
                'helpful_vote': int(score_data['helpful_votes'][i]),
                **placeholder_data
            }
            
            # Create review object
            review = AmazonReview(**review_data)
            reviews.append(review)
        
        return reviews


class ScoreAnalyzer:
    """
    Utility class for analyzing generated scores and their relationships.
    """
    
    @staticmethod
    def analyze_score_distribution(reviews: List[AmazonReview]) -> Dict[str, Any]:
        """
        Analyze the distribution of ratings and helpful votes.
        
        Args:
            reviews: List of generated reviews
            
        Returns:
            Dictionary with distribution statistics
        """
        ratings = [review.rating for review in reviews]
        helpful_votes = [review.helpful_vote for review in reviews]
        
        return {
            'rating_stats': {
                'mean': np.mean(ratings),
                'std': np.std(ratings),
                'distribution': {i: ratings.count(i) for i in range(1, 6)}
            },
            'helpful_vote_stats': {
                'mean': np.mean(helpful_votes),
                'std': np.std(helpful_votes),
                'max': max(helpful_votes),
                'min': min(helpful_votes)
            },
            'correlation': np.corrcoef(ratings, helpful_votes)[0, 1]
        }