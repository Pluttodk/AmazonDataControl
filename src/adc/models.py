from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class AmazonReview(BaseModel):
    """
    Pydantic model representing an Amazon product review.
    """
    rating: int = Field(..., ge=1, le=5, description="Review rating from 1 to 5 stars")
    title: str = Field(..., description="Review title")
    text: str = Field(..., description="Review text content")
    images: List[str] = Field(default_factory=list, description="List of image URLs attached to the review")
    asin: str = Field(..., description="Amazon Standard Identification Number (unique product identifier)")
    parent_asin: str = Field(..., description="Parent ASIN for product variants")
    user_id: str = Field(..., description="Unique identifier for the reviewer")
    timestamp: datetime = Field(..., description="Date and time when the review was posted")
    helpful_vote: int = Field(default=0, ge=0, description="Number of helpful votes the review received")
    verified_purchase: bool = Field(default=False, description="Whether the review is from a verified purchase")