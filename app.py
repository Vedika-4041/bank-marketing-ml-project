"""
Bank Marketing Classification - Interactive Streamlit Application
Author: BITS Pilani Student
Dataset: UCI Bank Marketing Dataset
Course: M.Tech (AIML/DSE) - Machine Learning Assignment 2
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Bank Marketing ML Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    </style>
    """, unsafe_allow_html=True)


def load_model(model_name):
    """
    Load the selected model from pickle file
    """
    model_files = {
        'Logistic Regression': 'models/logistic_regression_model.pkl',
        'Decision Tree': 'models/decision_tree_model.pkl',
        'K-Nearest Neighbors': 'models/knn_model.pkl',
        'Naive Bayes': 'models/naive_bayes_model.pkl',
        'Random Forest': 'models/random_forest_model.pkl',
        'XGBoost': 'models/xgboost_model.pkl'
    }

    try:
        with open(model_files[model_name], 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        st.error(f"Model file not found: {model_files[model_name]}")
        st.info("Please train the models first by running the individual model scripts.")
        return None


def preprocess_test_data(df, label_encoders, feature_names):
    """
    Preprocess the uploaded test data
    """
    data = df.copy()

    # Encode categorical variables
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing',
                       'loan', 'contact', 'month', 'poutcome']

    for col in categorical_cols:
        if col in data.columns:
            # Handle unseen categories
            le = label_encoders[col]
            data[col] = data[col].astype(str)
            data[col] = data[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Ensure all required features are present
    for feature in feature_names:
        if feature not in data.columns:
            data[feature] = 0

    # Select only the required features in the correct order
    X = data[feature_names]

    # Get target if present
    y = None
    if 'deposit' in df.columns:
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(df['deposit'])

    return X, y


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate all required evaluation metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics


def plot_confusion_matrix(cm, class_names=['No', 'Yes']):
    """
    Create an interactive confusion matrix heatmap
    """
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500,
        height=500,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )

    return fig


def plot_metrics_radar(metrics):
    """
    Create a radar chart for metrics visualization
    """
    categories = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Metrics',
        line=dict(color='#1f77b4', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Model Performance Radar Chart",
        width=600,
        height=500
    )

    return fig


def plot_metrics_bar(metrics):
    """
    Create a bar chart for metrics
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        marker=dict(
            color=list(metrics.values()),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Score")
        ),
        text=[f'{v:.4f}' for v in metrics.values()],
        textposition='outside'
    ))

    fig.update_layout(
        title="Performance Metrics Overview",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        height=500,
        showlegend=False
    )

    return fig


def main():
    """
    Main Streamlit application
    """
    # Header
    st.title("🏦 Bank Marketing Classification System")
    st.markdown("### ML Assignment 2 - BITS Pilani (WILP)")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")

    # Model selection
    st.sidebar.subheader("1️⃣ Select Model")
    model_options = [
        'Logistic Regression',
        'Decision Tree',
        'K-Nearest Neighbors',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
    ]
    selected_model = st.sidebar.selectbox(
        "Choose a classification model:",
        model_options,
        help="Select one of the 6 trained models"
    )

    # File upload
    st.sidebar.subheader("2️⃣ Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your test dataset in CSV format"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Instructions:**
    1. Select a classification model
    2. Upload test data (CSV format)
    3. View predictions and metrics

    **Required Columns:**
    age, job, marital, education, default, balance, housing, loan,
    contact, day, month, duration, campaign, pdays, previous, poutcome

    **Optional:** deposit (for evaluation)
    """)

    # Main content
    if uploaded_file is None:
        st.info("👈 Please upload a CSV file from the sidebar to begin analysis.")

        # Display sample format
        st.subheader("📋 Expected Data Format")
        sample_data = {
            'age': [30, 45, 25],
            'job': ['admin.', 'blue-collar', 'technician'],
            'marital': ['single', 'married', 'single'],
            'education': ['tertiary', 'secondary', 'tertiary'],
            'default': ['no', 'no', 'no'],
            'balance': [1500, 2000, 500],
            'housing': ['yes', 'yes', 'no'],
            'loan': ['no', 'no', 'yes'],
            'contact': ['cellular', 'unknown', 'cellular'],
            'day': [15, 20, 10],
            'month': ['may', 'jun', 'jul'],
            'duration': [300, 250, 400],
            'campaign': [1, 2, 1],
            'pdays': [-1, -1, -1],
            'previous': [0, 0, 0],
            'poutcome': ['unknown', 'unknown', 'unknown'],
            'deposit': ['yes', 'no', 'yes']
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

    else:
        # Load the uploaded file
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")

            # Display data preview
            with st.expander("🔍 View Uploaded Data (First 10 rows)", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            # Load selected model
            with st.spinner(f"Loading {selected_model} model..."):
                artifacts = load_model(selected_model)

            if artifacts is not None:
                # Extract model components
                model = artifacts['model']
                label_encoders = artifacts['label_encoders']
                feature_names = artifacts['feature_names']
                scaler = artifacts.get('scaler', None)

                # Preprocess data
                st.subheader("🔄 Preprocessing Data...")
                X_test, y_test = preprocess_test_data(df, label_encoders, feature_names)

                # Make predictions
                st.subheader(f"🎯 Running {selected_model}...")

                if scaler is not None:
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Display predictions
                st.success("✅ Predictions completed!")

                # Add predictions to dataframe
                results_df = df.copy()
                results_df['Predicted_Deposit'] = ['Yes' if p == 1 else 'No' for p in y_pred]
                results_df['Prediction_Probability'] = y_pred_proba

                with st.expander("📊 View Predictions", expanded=True):
                    st.dataframe(results_df, use_container_width=True)

                    # Download predictions
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Predictions as CSV",
                        data=csv,
                        file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )

                # If ground truth is available, calculate metrics
                if y_test is not None:
                    st.markdown("---")
                    st.subheader("📈 Model Evaluation Metrics")

                    # Calculate metrics
                    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

                    # Display metrics in columns
                    col1, col2, col3, col4, col5, col6 = st.columns(6)

                    with col1:
                        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    with col2:
                        st.metric("AUC", f"{metrics['AUC']:.4f}")
                    with col3:
                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                    with col4:
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                    with col5:
                        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                    with col6:
                        st.metric("MCC", f"{metrics['MCC']:.4f}")

                    st.markdown("---")

                    # Visualizations
                    viz_col1, viz_col2 = st.columns(2)

                    with viz_col1:
                        # Confusion Matrix
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm = plot_confusion_matrix(cm)
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with viz_col2:
                        # Metrics Radar Chart
                        fig_radar = plot_metrics_radar(metrics)
                        st.plotly_chart(fig_radar, use_container_width=True)

                    # Bar chart for metrics
                    fig_bar = plot_metrics_bar(metrics)
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Classification Report
                    st.markdown("---")
                    st.subheader("📋 Detailed Classification Report")
                    cr = classification_report(y_test, y_pred, target_names=['No', 'Yes'], output_dict=True)
                    cr_df = pd.DataFrame(cr).transpose()
                    st.dataframe(cr_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

                else:
                    st.warning("⚠️ No ground truth labels found in the uploaded data. Only predictions are shown.")

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format and required columns.")


# Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p><strong>Bank Marketing Classification System</strong></p>
        <p>M.Tech (AIML/DSE) - BITS Pilani (WILP) | Machine Learning Assignment 2</p>
        <p>Dataset: UCI Bank Marketing Dataset | Models: 6 Classification Algorithms</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
