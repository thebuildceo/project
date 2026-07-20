"""
Dataset Download Script

This script downloads the Telco Customer Churn dataset from Kaggle
and saves it to the data directory.
"""

import pandas as pd
import os


def download_dataset():
    """
    Download the Telco Customer Churn dataset.
    
    The dataset is loaded from a publicly available URL.
    """
    # Public URL for the Telco Customer Churn dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the dataset
    print("Downloading Telco Customer Churn dataset...")
    df = pd.read_csv(url)
    
    # Save to CSV
    filepath = os.path.join(data_dir, "telco_churn.csv")
    df.to_csv(filepath, index=False)
    
    print(f"Dataset downloaded and saved to {filepath}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst the few rows:")
    print(df.head())
    
    return filepath


if __name__ == "__main__":
    download_dataset()
