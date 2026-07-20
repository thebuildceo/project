"""
Model Training Script

This script trains the customer churn prediction model and saves it
for use in the Streamlit application.
"""

import os
import sys
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer


def train_and_save_model():
    """
    Train the churn prediction model and save it.
    """
    # Initialize processor and trainer
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Load data
    print("Loading dataset...")
    data_path = "data/genthropic_churn.csv"
    
    if not os.path.exists(data_path):
        print("Dataset not found. Please run download_data.py first.")
        return
    
    df = processor.load_data(data_path)
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    df_processed = processor.preprocess(df, fit=True, encode_target=True)
    
    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = processor.prepare_train_test_split(df_processed)
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Train model
    print("Training Random Forest model...")
    trainer.train_random_forest(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Display feature importance
    print("\n" + "="*50)
    print("TOP 10 FEATURE IMPORTANCE")
    print("="*50)
    feature_importance = trainer.get_feature_importance()
    print(feature_importance.head(10).to_string(index=False))
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = "models/churn_model.pkl"
    trainer.save_model(model_path)
    
    # Save processor for inference
    import joblib
    processor_path = "models/data_processor.pkl"
    joblib.dump(processor, processor_path)
    print(f"Data processor saved to {processor_path}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Model saved to: {model_path}")
    print(f"Processor saved to: {processor_path}")
    print("\nYou can now run the Streamlit application with:")
    print("streamlit run app.py")


if __name__ == "__main__":
    train_and_save_model()
