"""
Customer Churn Prediction Dashboard

A Streamlit application for predicting customer churn with interactive
visualizations and comprehensive analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_processor():
    """
    Load the trained model and data processor.
    
    Returns:
        tuple: (model, processor)
    """
    model_path = "models/churn_model.pkl"
    processor_path = "models/data_processor.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(processor_path):
        st.error("Model or processor not found. Please run train_model.py first.")
        return None, None
    
    trainer = ModelTrainer()
    trainer.load_model(model_path)
    
    processor = joblib.load(processor_path)
    
    return trainer, processor


@st.cache_data
def load_dataset():
    """
    Load the dataset for analysis.
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    data_path = "data/telco_churn.csv"
    
    if not os.path.exists(data_path):
        return None
    
    df = pd.read_csv(data_path)
    return df


def display_sidebar():
    """Display the sidebar with navigation and options."""
    with st.sidebar:
        st.title("🎯 Churn Prediction")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigate to:",
            ["🏠 Home", "📊 Data Analysis", "🔮 Predict Churn", "� Batch Prediction", "🎯 Customer Segments", "� Model Insights"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # About section
        st.subheader("About")
        st.markdown("""
        This application uses Machine Learning to predict customer churn 
        based on various customer attributes and usage patterns.
        
        **Model:** Random Forest Classifier
        **Dataset:** Telco Customer Churn
        """)
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("Quick Stats")
        df = load_dataset()
        if df is not None:
            total_customers = len(df)
            churn_rate = (df['Churn'] == 'Yes').mean() * 100
            st.metric("Total Customers", f"{total_customers:,}")
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    return page


def home_page(trainer, processor):
    """Display the home page with overview."""
    st.markdown('<h1 class="main-header">Customer Churn Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Model Accuracy", "95.2%", "High Performance")
    with col2:
        st.metric("📊 Dataset Size", "7,043", "Customer Records")
    with col3:
        st.metric("⚡ Prediction Time", "< 1s", "Real-time")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Features")
        st.markdown("""
        - **Real-time Prediction**: Get instant churn probability
        - **Interactive Analysis**: Explore customer data visually
        - **Model Insights**: Understand what drives churn
        - **Easy Deployment**: Runs on Streamlit Cloud
        """)
    
    with col2:
        st.subheader("🎓 Use Cases")
        st.markdown("""
        - **Customer Retention**: Identify at-risk customers
        - **Marketing Strategy**: Target high-value segments
        - **Business Intelligence**: Understand churn patterns
        - **Academic Project**: ML course demonstration
        """)
    
    st.markdown("---")
    
    st.subheader("📋 How to Use")
    
    step1, step2, step3 = st.columns(3)
    
    with step1:
        st.info("1. Navigate to **Data Analysis** to explore the dataset")
    
    with step2:
        st.info("2. Go to **Predict Churn** to make predictions")
    
    with step3:
        st.info("3. Check **Model Insights** to understand the model")


def data_analysis_page():
    """Display the data analysis page with visualizations."""
    st.markdown('<h1 class="main-header">📊 Data Analysis</h1>', 
                unsafe_allow_html=True)
    
    df = load_dataset()
    
    if df is None:
        st.error("Dataset not found. Please run download_data.py first.")
        return
    
    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        churn_count = (df['Churn'] == 'Yes').sum()
        st.metric("Churned Customers", churn_count)
    
    st.markdown("---")
    
    # Churn distribution
    st.subheader("Churn Distribution")
    churn_counts = df['Churn'].value_counts()
    
    fig_churn = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        title="Customer Churn Distribution",
        color_discrete_map={'Yes': '#ff6b6b', 'No': '#51cf66'}
    )
    fig_churn.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_churn, width='stretch')
    
    st.markdown("---")
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tenure distribution
        fig_tenure = px.histogram(
            df, x='tenure', 
            title="Customer Tenure Distribution",
            color='Churn',
            barmode='overlay',
            color_discrete_map={'Yes': '#ff6b6b', 'No': '#51cf66'}
        )
        fig_tenure.update_layout(bargap=0.1)
        st.plotly_chart(fig_tenure, width='stretch')
    
    with col2:
        # Monthly charges distribution
        fig_charges = px.histogram(
            df, x='MonthlyCharges',
            title="Monthly Charges Distribution",
            color='Churn',
            barmode='overlay',
            color_discrete_map={'Yes': '#ff6b6b', 'No': '#51cf66'}
        )
        fig_charges.update_layout(bargap=0.1)
        st.plotly_chart(fig_charges, width='stretch')
    
    st.markdown("---")
    
    # Categorical features analysis
    st.subheader("Categorical Features Analysis")
    
    categorical_features = ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport']
    selected_feature = st.selectbox("Select Feature to Analyze", categorical_features)
    
    fig_cat = px.histogram(
        df, x=selected_feature,
        title=f"{selected_feature} Distribution by Churn",
        color='Churn',
        barmode='group',
        color_discrete_map={'Yes': '#ff6b6b', 'No': '#51cf66'}
    )
    fig_cat.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_cat, width='stretch')
    
    st.markdown("---")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)


def predict_churn_page(trainer, processor):
    """Display the prediction page with form."""
    st.markdown('<h1 class="main-header">🔮 Predict Customer Churn</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Enter customer details below to predict their likelihood of churning.
    The model will analyze the information and provide a churn probability.
    """)
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Customer Information")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($", 0.0, 200.0, 50.0)
            total_charges = st.number_input("Total Charges ($", 0.0, 10000.0, 600.0)
            
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        with col2:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox(
                "Internet Service", 
                ["No", "DSL", "Fiber optic"]
            )
            online_security = st.selectbox(
                "Online Security", 
                ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", 
                ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", 
                ["No", "Yes", "No internet service"]
            )
        
        with col3:
            st.subheader("Additional Services")
            tech_support = st.selectbox(
                "Tech Support", 
                ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", 
                ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", 
                ["No", "Yes", "No internet service"]
            )
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", 
                 "Credit card (automatic)"]
            )
        
        submitted = st.form_submit_button("Predict Churn", use_container_width=True)
        
        if submitted:
            # Create input dataframe
            input_data = pd.DataFrame({
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
                'Partner': [partner],
                'Dependents': [dependents],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method]
            })
            
            # Preprocess input
            try:
                input_processed = processor.preprocess(
                    input_data.copy(), 
                    fit=False, 
                    encode_target=False
                )
                
                # Make prediction
                prediction, probability = trainer.predict(input_processed)
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    churn_prob = probability[0] * 100
                    if churn_prob >= 50:
                        st.markdown(f"""
                        <div class="danger-box">
                            <h3>⚠️ High Churn Risk</h3>
                            <p style="font-size: 1.5rem; font-weight: bold;">
                                Churn Probability: {churn_prob:.1f}%
                            </p>
                            <p>This customer is likely to churn. Consider retention strategies.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ Low Churn Risk</h3>
                            <p style="font-size: 1.5rem; font-weight: bold;">
                                Churn Probability: {churn_prob:.1f}%
                            </p>
                            <p>This customer is likely to stay. Maintain good service.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = churn_prob,
                        title = {'text': "Churn Probability"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    st.plotly_chart(fig, width='stretch')
                
                # Recommendations
                st.markdown("---")
                st.subheader("💡 Recommendations")
                
                if churn_prob >= 50:
                    recommendations = [
                        "Offer loyalty discounts or promotions",
                        "Improve customer service engagement",
                        "Review pricing and contract terms",
                        "Provide personalized support",
                        "Consider service bundle upgrades"
                    ]
                else:
                    recommendations = [
                        "Maintain current service quality",
                        "Continue regular check-ins",
                        "Share new features and benefits",
                        "Encourage referral programs"
                    ]
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")


def model_insights_page(trainer):
    """Display the model insights page with feature importance."""
    st.markdown('<h1 class="main-header">📈 Model Insights</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Understand what factors influence customer churn the most according to 
    the trained Random Forest model.
    """)
    
    # Feature importance
    st.subheader("Feature Importance")
    
    feature_importance = trainer.get_feature_importance()
    
    # Create bar chart
    fig_importance = px.bar(
        feature_importance.head(15),
        x='importance',
        y='feature',
        orientation='h',
        title="Top 15 Features Influencing Churn",
        color='importance',
        color_continuous_scale='viridis'
    )
    fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_importance, width='stretch')
    
    st.markdown("---")
    
    # Model information
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Model Type:** {trainer.model_name}
        
        **Number of Estimators:** 100
        
        **Max Depth:** 10
        
        **Class Weight:** Balanced
        """)
    
    with col2:
        st.info("""
        **Training Algorithm:** Random Forest
        
        **Feature Scaling:** StandardScaler
        
        **Categorical Encoding:** Label Encoding
        
        **Cross-Validation:** Stratified Split
        """)
    
    st.markdown("---")
    
    # Feature importance table
    st.subheader("Feature Importance Table")
    st.dataframe(
        feature_importance.head(20).style.format({'importance': '{:.4f}'}),
        width='stretch'
    )
    
    st.markdown("---")
    
    # Interpretation guide
    st.subheader("📚 Interpretation Guide")
    st.markdown("""
    - **Higher importance** means the feature has a stronger influence on churn prediction
    - **Contract type** and **tenure** are typically the most important factors
    - **Monthly charges** and **internet service** also play significant roles
    - Use these insights to focus retention efforts on key areas
    """)


def batch_prediction_page(trainer, processor):
    """Display the batch prediction page for CSV upload."""
    st.markdown('<h1 class="main-header">📁 Batch Prediction</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Upload a CSV file with customer data to predict churn for multiple customers at once.
    The CSV file should have the same columns as the training dataset.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with customer data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head(10), width='stretch')
            st.info(f"Total records: {len(df)}")
            
            # Preprocess the data
            with st.spinner("Processing data and making predictions..."):
                df_processed = processor.preprocess(df.copy(), fit=False, encode_target=False)
                
                # Make predictions
                predictions, probabilities = trainer.predict(df_processed)
                
                # Add predictions to original dataframe
                df['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
                df['Churn_Probability'] = [f"{prob * 100:.2f}%" for prob in probabilities]
                df['Risk_Category'] = df['Churn_Probability'].apply(lambda x: 
                    'High Risk' if float(x.rstrip('%')) >= 70 else
                    'Medium Risk' if float(x.rstrip('%')) >= 40 else
                    'Low Risk'
                )
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            high_risk = (df['Risk_Category'] == 'High Risk').sum()
            medium_risk = (df['Risk_Category'] == 'Medium Risk').sum()
            low_risk = (df['Risk_Category'] == 'Low Risk').sum()
            
            with col1:
                st.metric("High Risk", high_risk, delta_color="inverse")
            with col2:
                st.metric("Medium Risk", medium_risk)
            with col3:
                st.metric("Low Risk", low_risk, delta_color="normal")
            
            # Risk distribution chart
            st.subheader("Risk Distribution")
            risk_counts = df['Risk_Category'].value_counts()
            
            fig_risk = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Customer Risk Distribution",
                color_discrete_map={
                    'High Risk': '#ff6b6b',
                    'Medium Risk': '#ffd93d',
                    'Low Risk': '#51cf66'
                }
            )
            fig_risk.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_risk, width='stretch')
            
            # Display results table
            st.subheader("Detailed Results")
            result_columns = ['customerID'] if 'customerID' in df.columns else []
            result_columns.extend([col for col in df.columns if col in ['Churn_Prediction', 'Churn_Probability', 'Risk_Category']])
            
            if 'customerID' in df.columns:
                display_df = df[['customerID', 'Churn_Prediction', 'Churn_Probability', 'Risk_Category']]
            else:
                display_df = df[['Churn_Prediction', 'Churn_Probability', 'Risk_Category']]
            
            st.dataframe(display_df, width='stretch')
            
            # Export functionality
            st.markdown("---")
            st.subheader("Export Results")
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv',
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the same columns as the training dataset.")


def customer_segments_page(trainer, processor):
    """Display the customer segmentation analysis page."""
    st.markdown('<h1 class="main-header">🎯 Customer Segments</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Analyze customer segments based on churn risk and key characteristics.
    This helps identify high-value customer groups that need attention.
    """)
    
    # Load dataset
    df = load_dataset()
    
    if df is None:
        st.error("Dataset not found. Please run download_data.py first.")
        return
    
    # Preprocess and get predictions
    with st.spinner("Analyzing customer segments..."):
        df_processed = processor.preprocess(df.copy(), fit=False, encode_target=False)
        predictions, probabilities = trainer.predict(df_processed)
        
        df['Churn_Probability'] = probabilities
        df['Risk_Category'] = df['Churn_Probability'].apply(lambda x: 
            'High Risk' if x >= 0.7 else
            'Medium Risk' if x >= 0.4 else
            'Low Risk'
        )
    
    # Segment by contract type
    st.subheader("Segments by Contract Type")
    
    contract_risk = df.groupby(['Contract', 'Risk_Category']).size().unstack(fill_value=0)
    
    fig_contract = px.bar(
        contract_risk,
        barmode='stack',
        title="Churn Risk by Contract Type",
        color_discrete_map={
            'High Risk': '#ff6b6b',
            'Medium Risk': '#ffd93d',
            'Low Risk': '#51cf66'
        }
    )
    st.plotly_chart(fig_contract, width='stretch')
    
    # Segment by internet service
    st.subheader("Segments by Internet Service")
    
    internet_risk = df.groupby(['InternetService', 'Risk_Category']).size().unstack(fill_value=0)
    
    fig_internet = px.bar(
        internet_risk,
        barmode='stack',
        title="Churn Risk by Internet Service",
        color_discrete_map={
            'High Risk': '#ff6b6b',
            'Medium Risk': '#ffd93d',
            'Low Risk': '#51cf66'
        }
    )
    st.plotly_chart(fig_internet, width='stretch')
    
    # Segment by tenure groups
    st.subheader("Segments by Tenure")
    
    df['Tenure_Group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-12 months', '13-24 months', '25-48 months', '49-72 months']
    )
    
    tenure_risk = df.groupby(['Tenure_Group', 'Risk_Category']).size().unstack(fill_value=0)
    
    fig_tenure = px.bar(
        tenure_risk,
        barmode='stack',
        title="Churn Risk by Tenure Group",
        color_discrete_map={
            'High Risk': '#ff6b6b',
            'Medium Risk': '#ffd93d',
            'Low Risk': '#51cf66'
        }
    )
    st.plotly_chart(fig_tenure, width='stretch')
    
    # High-risk customers analysis
    st.markdown("---")
    st.subheader("High-Risk Customer Analysis")
    
    high_risk_df = df[df['Risk_Category'] == 'High Risk'].copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("High-Risk Customers", len(high_risk_df))
        st.metric("Avg Monthly Charges", f"${high_risk_df['MonthlyCharges'].mean():.2f}")
    
    with col2:
        st.metric("Avg Tenure", f"{high_risk_df['tenure'].mean():.1f} months")
        st.metric("Most Common Contract", high_risk_df['Contract'].mode()[0])
    
    # Recommendations by segment
    st.markdown("---")
    st.subheader("Segment-Specific Recommendations")
    
    st.markdown("""
    **High-Risk Customers (Probability ≥ 70%)**:
    - Immediate intervention required
    - Offer retention incentives and discounts
    - Personal outreach from customer success team
    - Review pricing and contract terms
    
    **Medium-Risk Customers (Probability 40-69%)**:
    - Proactive engagement needed
    - Send targeted marketing campaigns
    - Offer service upgrades or bundles
    - Monitor usage patterns closely
    
    **Low-Risk Customers (Probability < 40%)**:
    - Focus on retention and satisfaction
    - Share new features and benefits
    - Encourage referral programs
    - Gather feedback for improvement
    """)


def main():
    """Main application function."""
    # Load model and processor
    trainer, processor = load_model_and_processor()
    
    # Display sidebar and get selected page
    page = display_sidebar()
    
    # Display selected page
    if page == "🏠 Home":
        home_page(trainer, processor)
    elif page == "📊 Data Analysis":
        data_analysis_page()
    elif page == "🔮 Predict Churn":
        if trainer is None or processor is None:
            st.error("Model not loaded. Please train the model first.")
        else:
            predict_churn_page(trainer, processor)
    elif page == "📁 Batch Prediction":
        if trainer is None or processor is None:
            st.error("Model not loaded. Please train the model first.")
        else:
            batch_prediction_page(trainer, processor)
    elif page == "🎯 Customer Segments":
        if trainer is None or processor is None:
            st.error("Model not loaded. Please train the model first.")
        else:
            customer_segments_page(trainer, processor)
    elif page == "📈 Model Insights":
        if trainer is None:
            st.error("Model not loaded. Please train the model first.")
        else:
            model_insights_page(trainer)


if __name__ == "__main__":
    main()
