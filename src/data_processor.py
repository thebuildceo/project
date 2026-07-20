"""
Data Processor Module for Customer Churn Prediction

This module handles data loading, preprocessing, and feature engineering
for the Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataProcessor:
    """Handles data preprocessing and feature engineering."""
    
    def __init__(self):
        """Initialize the DataProcessor."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, filepath):
        """
        Load the customer churn dataset.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        df = pd.read_csv(filepath)
        return df
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and data types.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Convert TotalCharges to numeric, handling empty strings
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with median
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
        # Remove customer ID as it's not useful for prediction
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using label encoding.
        
        Args:
            df: Input dataframe
            fit: Whether to fit the encoders (True for training, False for inference)
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable if present
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
            
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col])
                else:
                    # Handle unseen categories by using a default value
                    df[col] = 0
                    
        return df
    
    def encode_target(self, df, fit=True):
        """
        Encode the target variable (Churn).
        
        Args:
            df: Input dataframe
            fit: Whether to fit the encoder
            
        Returns:
            pd.DataFrame: Dataframe with encoded target
        """
        if 'Churn' in df.columns:
            if fit:
                self.label_encoders['Churn'] = LabelEncoder()
                df['Churn'] = self.label_encoders['Churn'].fit_transform(df['Churn'])
            else:
                df['Churn'] = self.label_encoders['Churn'].transform(df['Churn'])
        return df
    
    def scale_features(self, df, fit=True):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input dataframe
            fit: Whether to fit the scaler
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variable if present
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')
            
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            
        return df
    
    def preprocess(self, df, fit=True, encode_target=True):
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input dataframe
            fit: Whether to fit transformers (True for training)
            encode_target: Whether to encode target variable
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Clean data
        df = self.clean_data(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Encode target if needed
        if encode_target:
            df = self.encode_target(df, fit=fit)
            
        # Scale numerical features
        df = self.scale_features(df, fit=fit)
        
        # Store feature columns
        if fit:
            self.feature_columns = [col for col in df.columns if col != 'Churn']
            
        return df
    
    def prepare_train_test_split(self, df, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            df: Preprocessed dataframe
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_data(self, df):
        """
        Get feature names and their original values for interpretation.
        
        Args:
            df: Original dataframe before preprocessing
            
        Returns:
            dict: Feature information dictionary
        """
        feature_info = {}
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'customerID' in categorical_cols:
            categorical_cols.remove('customerID')
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
            
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')
        if 'SeniorCitizen' in numerical_cols:
            numerical_cols.remove('SeniorCitizen')  # This is actually categorical
            
        feature_info['categorical'] = categorical_cols
        feature_info['numerical'] = numerical_cols
        
        return feature_info
