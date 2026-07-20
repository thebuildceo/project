# Customer Churn Prediction Dashboard

A Streamlit application for predicting customer churn using Machine Learning. This project demonstrates a complete ML pipeline with data preprocessing, model training, and interactive web-based predictions.

**Repository**: https://github.com/thebuildceo/project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview

This application uses a Random Forest classifier to predict customer churn probability based on various customer attributes including demographics, service subscriptions, and billing information. The model is trained on the publicly available Telco Customer Churn dataset and achieves approximately 76% accuracy with 84% ROC AUC.

### Key Features

- **Real-time Prediction**: Instant churn probability for individual customers
- **Batch Prediction**: Upload CSV files to predict churn for multiple customers at once
- **Interactive Data Analysis**: Explore customer data with visualizations
- **Customer Segmentation**: Analyze customer segments based on churn risk
- **Export Predictions**: Download prediction results as CSV files
- **Model Insights**: Understand feature importance and what drives churn

## Project Structure

```
customer_churn_prediction/
│
├── app.py                      # Main Streamlit application
├── train_model.py              # Model training script
├── download_data.py            # Dataset download script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_processor.py       # Data preprocessing module
│   └── model_trainer.py        # Model training module
│
├── data/                       # Dataset directory
│   └── telco_churn.csv         # Customer churn dataset
│
└── models/                     # Trained models directory
    ├── churn_model.pkl         # Trained Random Forest model
    └── data_processor.pkl      # Fitted data processor
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/thebuildceo/project.git
cd project
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download dataset:
```bash
python download_data.py
```

5. Train the model:
```bash
python train_model.py
```

## Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Application Pages

1. **Home Page**: View overview and quick statistics
2. **Data Analysis**: Explore the dataset with interactive visualizations
3. **Predict Churn**: Enter customer details to get churn predictions
4. **Batch Prediction**: Upload CSV files for bulk predictions
5. **Customer Segments**: Analyze customer segments by risk
6. **Model Insights**: Understand feature importance and model details

## Deployment on Streamlit Community Cloud

### Deployment Steps

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Sign up with your GitHub account
3. Click "New app"
4. Select repository: `thebuildceo/project`
5. Select branch: `main`
6. Set main file path to `app.py`
7. Click "Deploy"

Streamlit Cloud will automatically detect `requirements.txt` and install dependencies. The application will be live within a few minutes.

## Model Details

### Algorithm: Random Forest Classifier

**Parameters:**
- n_estimators: 100
- max_depth: 10
- min_samples_split: 10
- min_samples_leaf: 4
- class_weight: balanced
- random_state: 42

### Performance Metrics

- **Accuracy**: 75.7%
- **Precision**: 52.9%
- **Recall**: 77.3%
- **F1 Score**: 62.8%
- **ROC AUC**: 84.1%

### Feature Engineering

- **Categorical Encoding**: Label encoding for categorical variables
- **Numerical Scaling**: StandardScaler for numerical features
- **Missing Values**: Median imputation for TotalCharges
- **Feature Selection**: All features used for training

### Top Features Influencing Churn

1. Contract type (Month-to-month customers churn more)
2. Tenure (New customers are more likely to churn)
3. Monthly charges (Higher charges correlate with churn)
4. Internet service (Fiber optic users have higher churn)
5. Payment method (Electronic check users churn more)

## Dataset Information

**Dataset**: Telco Customer Churn

**Source**: IBM Telco Customer Churn Dataset

**Size**: 7,043 customer records

**Features**: 20 customer attributes including demographics, account information, services, and charges.

**Target**: Churn (Yes/No)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
