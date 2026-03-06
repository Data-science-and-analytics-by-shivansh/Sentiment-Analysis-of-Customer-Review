"""
Pytest Configuration and Shared Fixtures
========================================
Shared test fixtures and configuration for all tests

This file is automatically loaded by pytest
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_reviews():
    """Sample customer reviews for testing"""
    return [
        "This product is absolutely amazing! Best purchase ever.",
        "Terrible quality. Complete waste of money.",
        "It's okay. Nothing special but works as expected.",
        "Great value for the price. Highly recommended!",
        "Worst product I've ever bought. Do not buy.",
        "Average product. Could be better.",
        "Excellent quality and fast shipping!",
        "Poor customer service and low quality.",
        "Satisfied with the purchase. Good product.",
        "Not worth the money. Very disappointed."
    ]


@pytest.fixture
def sample_labels():
    """Sample sentiment labels"""
    return [
        "positive", "negative", "neutral",
        "positive", "negative", "neutral",
        "positive", "negative", "positive", "negative"
    ]


@pytest.fixture
def sample_dataframe(sample_reviews, sample_labels):
    """Sample DataFrame for testing"""
    return pd.DataFrame({
        'review_id': [f'REV-{i:05d}' for i in range(len(sample_reviews))],
        'review_text': sample_reviews,
        'sentiment': sample_labels,
        'rating': [5, 1, 3, 4, 1, 3, 5, 2, 4, 2],
        'product_line': ['Electronics'] * 5 + ['Home'] * 5,
        'timestamp': pd.date_range('2024-01-01', periods=len(sample_reviews), freq='D')
    })


@pytest.fixture
def large_dataset():
    """Generate larger dataset for performance testing"""
    np.random.seed(42)
    n_samples = 1000
    
    positive_templates = [
        "Excellent product",
        "Great quality",
        "Highly recommended",
        "Love it",
        "Amazing purchase"
    ]
    
    negative_templates = [
        "Terrible product",
        "Poor quality",
        "Waste of money",
        "Disappointed",
        "Do not buy"
    ]
    
    neutral_templates = [
        "It's okay",
        "Average product",
        "Nothing special",
        "Works fine",
        "Acceptable"
    ]
    
    reviews = []
    labels = []
    
    for i in range(n_samples):
        sentiment = np.random.choice(['positive', 'negative', 'neutral'])
        
        if sentiment == 'positive':
            review = np.random.choice(positive_templates)
        elif sentiment == 'negative':
            review = np.random.choice(negative_templates)
        else:
            review = np.random.choice(neutral_templates)
        
        reviews.append(review)
        labels.append(sentiment)
    
    return pd.DataFrame({
        'review_text': reviews,
        'sentiment': labels
    })


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    from sentiment_analysis import AnalysisConfig
    
    return AnalysisConfig(
        max_features=100,
        min_df=2,
        max_df=0.9,
        n_topics=5,
        test_size=0.2,
        random_state=42
    )


@pytest.fixture
def trained_classifier(mock_config, large_dataset):
    """Pre-trained classifier for testing"""
    from sentiment_analysis import SentimentClassifier
    
    classifier = SentimentClassifier(mock_config)
    
    texts = large_dataset['review_text'].tolist()
    labels = large_dataset['sentiment'].tolist()
    
    classifier.train(texts, labels)
    
    return classifier


@pytest.fixture
def mock_api_response():
    """Mock API response for testing"""
    return {
        'status': 'success',
        'predictions': [
            {
                'text': 'Great product',
                'sentiment': 'positive',
                'confidence': 0.95
            }
        ]
    }


# Pytest hooks
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers automatically based on test names
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "slow" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# Assertion helper
class Helpers:
    """Helper methods for tests"""
    
    @staticmethod
    def is_valid_sentiment(sentiment):
        """Check if sentiment is valid"""
        return sentiment in ['positive', 'negative', 'neutral']
    
    @staticmethod
    def is_probability_distribution(probs):
        """Check if array is a valid probability distribution"""
        return np.allclose(probs.sum(), 1.0) and np.all(probs >= 0) and np.all(probs <= 1)


@pytest.fixture
def helpers():
    """Provide helper methods to tests"""
    return Helpers()
