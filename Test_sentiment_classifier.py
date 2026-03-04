"""
Unit Tests for Sentiment Classifier Module
==========================================
Tests for sentiment_analysis.py SentimentClassifier class

Run with: pytest tests/test_sentiment_classifier.py -v
"""

import pytest
import pandas as pd
import numpy as np
from sentiment_analysis import (
    SentimentClassifier,
    AnalysisConfig
)


class TestSentimentClassifier:
    """Test suite for SentimentClassifier class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return AnalysisConfig(
            max_features=100,
            test_size=0.2,
            random_state=42
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        texts = [
            "This product is excellent and amazing",
            "Great quality, highly recommend",
            "Wonderful experience, love it",
            "Terrible product, complete waste",
            "Poor quality, very disappointed",
            "Worst purchase ever made",
            "It's okay, nothing special",
            "Average product, works fine"
        ]
        
        labels = [
            "positive", "positive", "positive",
            "negative", "negative", "negative",
            "neutral", "neutral"
        ]
        
        return texts, labels
    
    def test_classifier_initialization(self, config):
        """Test classifier initialization"""
        classifier = SentimentClassifier(config)
        
        assert classifier.config == config
        assert classifier.tfidf_vectorizer is not None
        assert classifier.classifier is not None
        assert classifier.scaler is not None
        assert not classifier.is_fitted
    
    def test_train_basic(self, config, sample_data):
        """Test basic training"""
        classifier = SentimentClassifier(config)
        texts, labels = sample_data
        
        metrics = classifier.train(texts, labels)
        
        assert classifier.is_fitted
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_train_with_linguistic_features(self, config, sample_data):
        """Test training with linguistic features"""
        classifier = SentimentClassifier(config)
        texts, labels = sample_data
        
        # Create dummy linguistic features
        linguistic_features = np.random.rand(len(texts), 5)
        
        metrics = classifier.train(texts, labels, linguistic_features)
        
        assert classifier.is_fitted
        assert 'accuracy' in metrics
    
    def test_train_insufficient_data(self, config):
        """Test training with insufficient data"""
        classifier = SentimentClassifier(config)
        
        texts = ["Good product"]
        labels = ["positive"]
        
        with pytest.raises(ValueError):
            classifier.train(texts, labels)
    
    def test_predict_before_training(self, config):
        """Test prediction before training raises error"""
        classifier = SentimentClassifier(config)
        
        with pytest.raises(ValueError, match="must be trained first"):
            classifier.predict(["test text"])
    
    def test_predict_basic(self, config, sample_data):
        """Test basic prediction"""
        classifier = SentimentClassifier(config)
        texts, labels = sample_data
        
        # Train
        classifier.train(texts, labels)
        
        # Predict
        test_texts = ["Great product", "Terrible quality"]
        pred_labels, pred_probs = classifier.predict(test_texts)
        
        assert len(pred_labels) == 2
        assert len(pred_probs) == 2
        assert pred_labels[0] in ['positive', 'negative', 'neutral']
        assert pred_probs.shape[0] == 2
    
    def test_predict_probabilities_sum_to_one(self, config, sample_data):
        """Test that prediction probabilities sum to 1"""
        classifier = SentimentClassifier(config)
        texts, labels = sample_data
        
        classifier.train(texts, labels)
        
        _, pred_probs = classifier.predict(["test text"])
        
        prob_sum = pred_probs[0].sum()
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_predict_confidence(self, config, sample_data):
        """Test prediction confidence scores"""
        classifier = SentimentClassifier(config)
        texts, labels = sample_data
        
        classifier.train(texts, labels)
        
        # Strong positive text should have high confidence
        _, probs = classifier.predict(["Absolutely amazing excellent wonderful"])
        max_prob = probs[0].max()
        
        assert max_prob > 0.5  # Should be confident
    
    def test_cross_validation_scores(self, config, sample_data):
        """Test cross-validation scoring"""
        classifier = SentimentClassifier(config)
        texts, labels = sample_data
        
        metrics = classifier.train(texts, labels)
        
        assert 'cv_scores' in metrics
        assert 'cv_mean' in metrics
        assert 'cv_std' in metrics
        assert len(metrics['cv_scores']) == min(config.cv_folds, 5)
    
    def test_feature_importance(self, config, sample_data):
        """Test feature importance extraction"""
        config.model_type = "logistic"  # Logistic has feature importance
        classifier = SentimentClassifier(config)
        texts, labels = sample_data
        
        metrics = classifier.train(texts, labels)
        
        if 'feature_importance' in metrics:
            assert isinstance(metrics['feature_importance'], dict)
    
    def test_different_model_types(self, sample_data):
        """Test different classifier types"""
        texts, labels = sample_data
        
        for model_type in ['logistic', 'nb', 'rf']:
            config = AnalysisConfig(model_type=model_type, max_features=50)
            classifier = SentimentClassifier(config)
            
            try:
                metrics = classifier.train(texts, labels)
                assert classifier.is_fitted
                assert 'accuracy' in metrics
            except Exception as e:
                pytest.fail(f"Model type {model_type} failed: {e}")
    
    def test_empty_text_prediction(self, config, sample_data):
        """Test prediction on empty text"""
        classifier = SentimentClassifier(config)
        texts, labels = sample_data
        
        classifier.train(texts, labels)
        
        pred_labels, pred_probs = classifier.predict([""])
        
        assert len(pred_labels) == 1
        assert len(pred_probs) == 1
    
    def test_special_characters_prediction(self, config, sample_data):
        """Test prediction on text with special characters"""
        classifier = SentimentClassifier(config)
        texts, labels = sample_data
        
        classifier.train(texts, labels)
        
        test_text = "Great!!! Amazing@@@ #Best $$$"
        pred_labels, _ = classifier.predict([test_text])
        
        assert pred_labels[0] in ['positive', 'negative', 'neutral']
    
    def test_batch_prediction(self, config, sample_data):
        """Test batch prediction"""
        classifier = SentimentClassifier(config)
        texts, labels = sample_data
        
        classifier.train(texts, labels)
        
        test_texts = ["Great"] * 100
        pred_labels, pred_probs = classifier.predict(test_texts)
        
        assert len(pred_labels) == 100
        assert pred_probs.shape[0] == 100


class TestSentimentClassifierIntegration:
    """Integration tests for sentiment classifier"""
    
    def test_end_to_end_workflow(self):
        """Test complete training and prediction workflow"""
        # Configuration
        config = AnalysisConfig(max_features=100)
        
        # Data
        train_texts = [
            "Excellent product, highly recommended",
            "Great quality, very satisfied",
            "Terrible waste of money",
            "Poor quality, very disappointed",
            "It's okay, nothing special"
        ] * 10  # Repeat for sufficient data
        
        train_labels = (
            ["positive"] * 2 + 
            ["negative"] * 2 + 
            ["neutral"]
        ) * 10
        
        # Train
        classifier = SentimentClassifier(config)
        metrics = classifier.train(train_texts, train_labels)
        
        # Verify training
        assert classifier.is_fitted
        assert metrics['accuracy'] > 0.5  # Should be better than random
        
        # Predict
        test_texts = [
            "Amazing product",
            "Awful quality",
            "Okay product"
        ]
        
        pred_labels, pred_probs = classifier.predict(test_texts)
        
        # Verify predictions make sense
        assert pred_labels[0] == "positive"  # Amazing should be positive
        assert pred_labels[1] == "negative"  # Awful should be negative
    
    def test_model_persistence(self, tmp_path):
        """Test saving and loading classifier"""
        import pickle
        
        config = AnalysisConfig()
        classifier = SentimentClassifier(config)
        
        texts = ["Great product"] * 20 + ["Bad product"] * 20
        labels = ["positive"] * 20 + ["negative"] * 20
        
        classifier.train(texts, labels)
        
        # Save
        model_file = tmp_path / "classifier.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(classifier, f)
        
        # Load
        with open(model_file, 'rb') as f:
            loaded_classifier = pickle.load(f)
        
        # Test loaded model
        pred_labels, _ = loaded_classifier.predict(["Great product"])
        assert pred_labels[0] == "positive"
    
    def test_consistent_predictions(self):
        """Test that predictions are consistent"""
        config = AnalysisConfig(random_state=42)
        
        texts = ["Great product"] * 20 + ["Bad product"] * 20
        labels = ["positive"] * 20 + ["negative"] * 20
        
        # Train first classifier
        classifier1 = SentimentClassifier(config)
        classifier1.train(texts, labels)
        pred1, _ = classifier1.predict(["Test product"])
        
        # Train second classifier with same config
        classifier2 = SentimentClassifier(config)
        classifier2.train(texts, labels)
        pred2, _ = classifier2.predict(["Test product"])
        
        # Should produce same results
        assert pred1[0] == pred2[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
