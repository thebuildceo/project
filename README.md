# Customer Churn Prediction Dashboard

A production-ready Streamlit application for predicting customer churn using Machine Learning. This project demonstrates a complete ML pipeline with data preprocessing, model training, and interactive web-based predictions.

**Repository**: https://github.com/thebuildceo/project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This application uses a Random Forest classifier to predict customer churn probability based on various customer attributes including demographics, service subscriptions, and billing information. The model is trained on the publicly available Telco Customer Churn dataset and achieves approximately 76% accuracy with 84% ROC AUC.

### Key Features

- **Real-time Prediction**: Get instant churn probability for individual customers
- **Batch Prediction**: Upload CSV files to predict churn for multiple customers at once
- **Interactive Data Analysis**: Explore customer data with beautiful visualizations
- **Customer Segmentation**: Analyze customer segments based on churn risk
- **Export Predictions**: Download prediction results as CSV files
- **Model Insights**: Understand feature importance and what drives churn
- **Production-Ready**: Clean, modular, and well-documented code
- **Cloud Deployable**: Runs seamlessly on Streamlit Community Cloud

## 🏗️ Project Structure

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
├── models/                     # Trained models directory
│   ├── churn_model.pkl         # Trained Random Forest model
│   └── data_processor.pkl      # Fitted data processor
│
└── screenshots/                # Application screenshots
    ├── home_page.png
    ├── data_analysis.png
    ├── prediction.png
    └── model_insights.png
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

```bash
python download_data.py
```

This will download the Telco Customer Churn dataset to the `data/` directory.

### Step 5: Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a Random Forest classifier
- Evaluate the model performance
- Save the trained model and data processor to the `models/` directory

## 🎮 Running the Application

### Local Development

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Home Page**: View overview and quick statistics
2. **Data Analysis**: Explore the dataset with interactive visualizations
3. **Predict Churn**: Enter customer details to get churn predictions
4. **Model Insights**: Understand feature importance and model details

## 🌐 Deployment on Streamlit Community Cloud

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/customer-churn-prediction.git
git push -u origin main
```

### Step 2: Create Streamlit Cloud Account

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Sign up with your GitHub account
3. Click "New app"

### Step 3: Deploy the Application

1. Select your GitHub repository
2. Select the `main` branch
3. Set the main file path to `app.py`
4. Click "Deploy"

### Step 4: Configure Requirements

Streamlit Cloud will automatically detect `requirements.txt` and install dependencies. The application will be live within a few minutes.

### Important Notes for Deployment

- The `data/` and `models/` directories should be included in your repository
- Ensure `requirements.txt` is in the root directory
- The application will automatically download the dataset on first run if needed
- No API keys or external services are required

## 📊 Model Details

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

 Top Features Influencing Churn:

1. Contract type (Month-to-month customers churn more)
2. Tenure (New customers are more likely to churn)
3. Monthly charges (Higher charges correlate with churn)
4. Internet service (Fiber optic users have higher churn)
5. Payment method (Electronic check users churn more)

## 🧪 Testing the Model

To test the model with new data:

```python
import pandas as pd
import joblib
from src.data_processor import DataProcessor

# Load model and processor
trainer = ModelTrainer()
trainer.load_model('models/churn_model.pkl')
processor = joblib.load('models/data_processor.pkl')

# Create sample input
input_data = pd.DataFrame({
    'tenure': [12],
    'MonthlyCharges': [50.0],
    'TotalCharges': [600.0],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['No'],
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': ['DSL'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check']
})

# Preprocess and predict
input_processed = processor.preprocess(input_data.copy(), fit=False, encode_target=False)
prediction, probability = trainer.predict(input_processed)

print(f"Churn Probability: {probability[0] * 100:.1f}%")
```

## 📚 Dataset Information

**Dataset**: Telco Customer Churn

**Source**: IBM Telco Customer Churn Dataset

**Size**: 7,043 customer records

**Features**: 20 customer attributes including:
- Demographics (SeniorCitizen, Partner, Dependents)
- Account information (Tenure, Contract, PaperlessBilling, PaymentMethod)
- Services (PhoneService, MultipleLines, InternetService, OnlineSecurity, etc.)
- Charges (MonthlyCharges, TotalCharges)

**Target**: Churn (Yes/No)

**License**: The dataset is publicly available for educational purposes.

## 🔧 Customization

### Changing the Model

Edit `train_model.py` to use different algorithms:

```python
# For Gradient Boosting
trainer.train_gradient_boosting(X_train, y_train)

# For Logistic Regression
trainer.train_logistic_regression(X_train, y_train)
```

### Adding New Features

Modify `src/data_processor.py` to add custom feature engineering:

```python
def add_custom_features(self, df):
    df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    return df
```

### Modifying the UI

Edit `app.py` to customize the Streamlit interface:
- Add new pages in the sidebar
- Modify input forms
- Change visualizations
- Add new charts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Built as a college assignment for an Artificial Intelligence and Machine Learning course.

## 🙏 Acknowledgments

- IBM for the Telco Customer Churn dataset
- Streamlit team for the amazing framework
- scikit-learn team for the ML library
- Plotly team for interactive visualizations

## 📞 Support

For issues or questions, please open an issue on GitHub.

## 🎓 Educational Value

This project demonstrates:

- **Data Preprocessing**: Handling missing values, encoding categorical variables, scaling features
- **Model Training**: Using scikit-learn for ML model development
- **Model Evaluation**: Understanding accuracy, precision, recall, F1-score, and ROC AUC
- **Feature Importance**: Interpreting ML models and understanding feature impact
- **Web Development**: Building interactive applications with Streamlit
- **Deployment**: Deploying ML applications to the cloud
- **Software Engineering**: Writing modular, maintainable, and documented code

Perfect for ML course assignments, portfolio projects, and learning production ML workflows.
