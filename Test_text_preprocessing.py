"""
Unit Tests for Text Preprocessing Module
========================================
Tests for sentiment_analysis.py TextPreprocessor class

Run with: pytest tests/test_text_preprocessing.py -v
"""

import pytest
import pandas as pd
import numpy as np
from sentiment_analysis import (
    TextPreprocessor,
    AnalysisConfig
)


class TestTextPreprocessor:
    """Test suite for TextPreprocessor class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return AnalysisConfig()
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return TextPreprocessor()
    
    def test_clean_text_basic(self, preprocessor, config):
        """Test basic text cleaning"""
        text = "This is a GREAT product!"
        cleaned = preprocessor.clean_text(text, config)
        
        assert cleaned == "this is a great product"
        assert cleaned.islower()
        assert "!" not in cleaned
    
    def test_clean_text_urls(self, preprocessor, config):
        """Test URL removal"""
        text = "Check out http://example.com for more info"
        cleaned = preprocessor.clean_text(text, config)
        
        assert "http" not in cleaned
        assert "example.com" not in cleaned
    
    def test_clean_text_emails(self, preprocessor, config):
        """Test email removal"""
        text = "Contact us at support@example.com"
        cleaned = preprocessor.clean_text(text, config)
        
        assert "@" not in cleaned
        assert "support" not in cleaned or "example" not in cleaned
    
    def test_clean_text_html(self, preprocessor, config):
        """Test HTML tag removal"""
        text = "<p>Great product</p> <b>highly recommended</b>"
        cleaned = preprocessor.clean_text(text, config)
        
        assert "<p>" not in cleaned
        assert "</p>" not in cleaned
        assert "<b>" not in cleaned
    
    def test_clean_text_contractions(self, preprocessor, config):
        """Test contraction expansion"""
        text = "I can't believe it's so good"
        cleaned = preprocessor.clean_text(text, config)
        
        assert "cannot" in cleaned or "can not" in cleaned
        assert "can't" not in cleaned
    
    def test_clean_text_empty_input(self, preprocessor, config):
        """Test handling of empty input"""
        assert preprocessor.clean_text("", config) == ""
        assert preprocessor.clean_text(None, config) == ""
    
    def test_clean_text_special_characters(self, preprocessor, config):
        """Test special character handling"""
        text = "Product@#$% is great!!!"
        cleaned = preprocessor.clean_text(text, config)
        
        assert "@#$%" not in cleaned
        assert "!" not in cleaned
    
    def test_tokenize_basic(self, preprocessor, config):
        """Test basic tokenization"""
        text = "This is a great product"
        tokens = preprocessor.tokenize(text, config)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "great" in tokens
        assert "product" in tokens
    
    def test_tokenize_stopwords(self, preprocessor, config):
        """Test stopword removal"""
        config.remove_stopwords = True
        text = "this is the best product"
        tokens = preprocessor.tokenize(text, config)
        
        assert "best" in tokens
        assert "product" in tokens
        # Stopwords should be removed
        assert "the" not in tokens
        assert "is" not in tokens
    
    def test_tokenize_numbers(self, preprocessor, config):
        """Test number removal option"""
        config.remove_numbers = True
        text = "Product 123 is great"
        tokens = preprocessor.tokenize(text, config)
        
        assert "123" not in tokens
        assert "great" in tokens
    
    def test_tokenize_length_filter(self, preprocessor, config):
        """Test token length filtering"""
        config.min_word_length = 3
        config.max_word_length = 10
        text = "a product is excellent"
        tokens = preprocessor.tokenize(text, config)
        
        assert "a" not in tokens  # Too short
        assert "product" in tokens
        assert "excellent" in tokens
    
    def test_extract_ngrams(self, preprocessor):
        """Test n-gram extraction"""
        tokens = ["great", "product", "highly", "recommended"]
        bigrams = preprocessor.extract_ngrams(tokens, n=2)
        
        assert "great product" in bigrams
        assert "product highly" in bigrams
        assert "highly recommended" in bigrams
        assert len(bigrams) == 3
    
    def test_extract_ngrams_insufficient_tokens(self, preprocessor):
        """Test n-gram extraction with insufficient tokens"""
        tokens = ["great"]
        bigrams = preprocessor.extract_ngrams(tokens, n=2)
        
        assert len(bigrams) == 0
    
    def test_calculate_sentiment_features(self, preprocessor):
        """Test sentiment feature calculation"""
        text = "This product is excellent! Highly recommended!!!"
        features = preprocessor.calculate_sentiment_features(text)
        
        assert isinstance(features, dict)
        assert 'word_count' in features
        assert 'positive_word_count' in features
        assert 'negative_word_count' in features
        assert 'exclamation_count' in features
        
        assert features['word_count'] > 0
        assert features['exclamation_count'] == 3
        assert features['positive_word_count'] > 0
    
    def test_calculate_sentiment_features_negative(self, preprocessor):
        """Test sentiment features for negative text"""
        text = "Terrible product, complete waste of money"
        features = preprocessor.calculate_sentiment_features(text)
        
        assert features['negative_word_count'] > 0
        assert features['sentiment_ratio'] < 0
    
    def test_calculate_sentiment_features_neutral(self, preprocessor):
        """Test sentiment features for neutral text"""
        text = "The product arrived on time"
        features = preprocessor.calculate_sentiment_features(text)
        
        # Should have minimal sentiment words
        assert features['positive_word_count'] + features['negative_word_count'] <= 1
    
    def test_calculate_sentiment_features_empty(self, preprocessor):
        """Test sentiment features for empty text"""
        features = preprocessor.calculate_sentiment_features("")
        
        assert features['word_count'] == 0
        assert features['sentiment_ratio'] == 0.0
    
    def test_negation_detection(self, preprocessor):
        """Test negation word detection"""
        text = "This product is not good, never buy it"
        features = preprocessor.calculate_sentiment_features(text)
        
        assert features['negation_count'] >= 2
    
    def test_intensifier_detection(self, preprocessor):
        """Test intensifier word detection"""
        text = "This is very extremely absolutely amazing"
        features = preprocessor.calculate_sentiment_features(text)
        
        assert features['intensifier_count'] >= 3


class TestTextPreprocessorIntegration:
    """Integration tests for text preprocessing pipeline"""
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing workflow"""
        config = AnalysisConfig()
        preprocessor = TextPreprocessor()
        
        # Sample review
        text = """
        <p>This product is AMAZING! I can't believe how good it is.
        Check out http://example.com for more info.
        Contact: support@example.com
        Product #123 is the best!!!
        </p>
        """
        
        # Clean
        cleaned = preprocessor.clean_text(text, config)
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
        
        # Tokenize
        tokens = preprocessor.tokenize(cleaned, config)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        # Extract features
        features = preprocessor.calculate_sentiment_features(text)
        assert isinstance(features, dict)
        assert features['word_count'] > 0
        assert features['positive_word_count'] > 0
    
    def test_batch_processing(self):
        """Test preprocessing multiple texts"""
        config = AnalysisConfig()
        preprocessor = TextPreprocessor()
        
        texts = [
            "Great product!",
            "Terrible quality",
            "It's okay, nothing special"
        ]
        
        cleaned_texts = [preprocessor.clean_text(t, config) for t in texts]
        
        assert len(cleaned_texts) == 3
        assert all(isinstance(t, str) for t in cleaned_texts)
        assert all(t.islower() for t in cleaned_texts)
    
    def test_feature_consistency(self):
        """Test that features are consistent across runs"""
        preprocessor = TextPreprocessor()
        text = "This is a great product"
        
        features1 = preprocessor.calculate_sentiment_features(text)
        features2 = preprocessor.calculate_sentiment_features(text)
        
        assert features1 == features2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
