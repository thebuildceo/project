"""
Model Trainer Module for Customer Churn Prediction

This module handles model training, evaluation, and inference
for predicting customer churn.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self):
        """Initialize the ModelTrainer."""
        self.model = None
        self.model_name = None
        self.feature_importance = None
        
    def train_random_forest(self, X_train, y_train, **params):
        """
        Train a Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **params: Additional parameters for the model
            
        Returns:
            RandomForestClassifier: Trained model
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        default_params.update(params)
        
        self.model = RandomForestClassifier(**default_params)
        self.model_name = 'Random Forest'
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def train_gradient_boosting(self, X_train, y_train, **params):
        """
        Train a Gradient Boosting classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **params: Additional parameters for the model
            
        Returns:
            GradientBoostingClassifier: Trained model
        """
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        }
        default_params.update(params)
        
        self.model = GradientBoostingClassifier(**default_params)
        self.model_name = 'Gradient Boosting'
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def train_logistic_regression(self, X_train, y_train, **params):
        """
        Train a Logistic Regression classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **params: Additional parameters for the model
            
        Returns:
            LogisticRegression: Trained model
        """
        default_params = {
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        default_params.update(params)
        
        self.model = LogisticRegression(**default_params)
        self.model_name = 'Logistic Regression'
        self.model.fit(X_train, y_train)
        
        # Store feature importance (coefficients)
        if hasattr(self.model, 'coef_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            tuple: (predictions, probabilities)
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            The loaded model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_importance = model_data['feature_importance']
        print(f"Model loaded from {filepath}")
        return self.model
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        return self.feature_importance
