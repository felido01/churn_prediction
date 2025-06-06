import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Customer Churn Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for professional styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header { 
        font-size: 42px; 
        color: #FFFFFF; 
        font-weight: 700; 
        text-align: center; 
        margin-bottom: 20px; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header { 
        font-size: 28px; 
        color: #2DD4BF; 
        font-weight: 600; 
        margin: 30px 0 20px; 
        text-align: center;
    }
    .metric-box { 
        background: #0F172A;
        padding: 20px; 
        border-radius: 12px; 
        text-align: center; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-in;
    }
    . necro-box:hover { 
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    .metric-box.churn { border-left: 4px solid #F43F5E; }
    .metric-box.customers { border-left: 4px solid #2DD4BF; }
    .metric-box.tenure { border-left: 4px solid #2DD4BF; }
    .metric-box.charges { border-left: 4px solid #2DD4BF; }
    .metric-box.senior { border-left: 4px solid #2DD4BF; }
    .stButton>button { 
        background: #F3F4F6; 
        color: #0F172A; 
        border-radius: 8px; 
        padding: 12px 24px; 
        font-size: 16px;
        font-weight: 500;
        border: 1px solid #2DD4BF;
        transition: background 0.3s ease, transform 0.3s ease;
    }
    .stButton>button:hover { 
        background: #2DD4BF; 
        color: #FFFFFF;
        transform: scale(1.05);
    }
    .stSelectbox div[data-baseweb="select"]>div {
        background: #F3F4F6;
        color: #0F172A;
        border: 1px solid #2DD4BF;
        border-radius: 8px;
    }
    .stSelectbox div[data-baseweb="select"]>div:hover {
        background: #E5E7EB;
    }
    .stSlider div[role="slider"] {
        background: #2DD4BF;
    }
    .hero-section {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.8), rgba(15, 23, 42, 0.7)), 
                    url('https://images.unsplash.com/photo-1551288049-b1f3a85551d4?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80') center/cover;
        padding: 60px 20px;
        border-radius: 15px;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 30px;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    .hero-text {
        font-size: 20px;
        color: #D1D5DB;
        max-width: 600px;
        margin: 0 auto 20px;
    }
    .mission-section {
        background: #0F172A;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin: 20px 0;
        border: 1px solid #2DD4BF;
    }
    .mission-section h3 {
        font-size: 24px;
        color: #FFFFFF;
        margin-bottom: 15px;
    }
    .mission-section p {
        font-size: 16px;
        color: #D1D5DB;
        line-height: 1.6;
        margin-bottom: 15px;
    }
    .preview-section {
        background: #1E293B;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin: 20px 0;
        border: 1px solid #2DD4BF;
    }
    .preview-section h3 {
        font-size: 24px;
        color: #FFFFFF;
        margin-bottom: 15px;
    }
    .footer-section {
        background: #0F172A;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 30px;
        border-top: 1px solid #2DD4BF;
    }
    .footer-section p {
        font-size: 14px;
        color: #D1D5DB;
        margin: 5px 0;
    }
    .footer-section a {
        color: #2DD4BF;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    .footer-section a:hover {
        color: #F43F5E;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0F172A, #1E293B);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .sidebar-header {
        font-size: 24px;
        font-weight: 600;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 10px;
    }
    .sidebar-subtext {
        font-size: 14px;
        color: #D1D5DB;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar-nav-item {
        display: flex;
        align-items: center;
        padding: 12px 15px;
        margin: 5px 0;
        border-radius: 8px;
        font-size: 16px;
        color: #FFFFFF;
        transition: background 0.3s ease, transform 0.2s ease;
        cursor: pointer;
    }
    .sidebar-nav-item:hover {
        background: #2DD4BF;
        color: #0F172A;
        transform: translateX(5px);
    }
    .sidebar-nav-item.active {
        background: #2DD4BF;
        color: #0F172A;
        font-weight: 600;
        border-left: 4px solid #F43F5E;
    }
    .sidebar-nav-item img {
        margin-right: 10px;
        width: 24px;
        height: 24px;
    }
    .sidebar-logo {
        display: block;
        margin: 0 auto 20px;
        width: 100px;
    }
    .collapsible-section {
        margin-top: 20px;
        padding: 10px;
        border-radius: 8px;
        background: #1E293B;
    }
    .collapsible-section h4 {
        font-size: 16px;
        color: #FFFFFF;
        margin-bottom: 10px;
    }
    .collapsible-section p {
        font-size: 14px;
        color: #D1D5DB;
        margin: 0;
    }
    .about-dataset {
        background: #0F172A;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin: 20px 0;
        border: 1px solid #2DD4BF;
    }
    .about-dataset h3 {
        font-size: 24px;
        color: #FFFFFF;
        margin-bottom: 15px;
    }
    .about-dataset p {
        font-size: 16px;
        color: #D1D5DB;
        line-height: 1.6;
        margin-bottom: 15px;
    }
    .about-dataset ul {
        list-style: none;
        padding: 0;
    }
    .about-dataset li {
        display: flex;
        align-items: center;
        font-size: 15px;
        color: #D1D5DB;
        margin: 10px 0;
    }
    .about-dataset li::before {
        content: '•';
        color: #2DD4BF;
        margin-right: 10px;
        font-size: 18px;
    }
    .project-goal {
        background: #0F172A;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin: 20px 0;
        border: 1px solid #2DD4BF;
    }
    .project-goal h3 {
        font-size: 24px;
        color: #FFFFFF;
        margin-bottom: 15px;
    }
    .project-goal p {
        font-size: 16px;
        color: #D1D5DB;
        line-height: 1.6;
        margin-bottom: 15px;
    }
    .filter-container {
        background: #1E293B;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #2DD4BF;
    }
    .filter-container h4 {
        font-size: 18px;
        color: #FFFFFF;
        margin-bottom: 15px;
    }
    .chart-card {
        background: #1E293B;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin: 10px 0;
    }
    .contact-links {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 15px;
    }
    .contact-links a {
        color: #2DD4BF;
        font-size: 24px;
        transition: color 0.3s ease, transform 0.3s ease;
    }
    .contact-links a:hover {
        color: #F43F5E;
        transform: scale(1.2);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .sidebar-footer {
        margin-top: 20px;
        text-align: center;
        font-size: 12px;
        color: #D1D5DB;
        border-top: 1px solid #2DD4BF;
        padding-top: 10px;
    }
    @media (max-width: 768px) {
        .main-header { font-size: 32px; }
        .hero-text { font-size: 18px; }
        .sub-header { font-size: 24px; }
        .metric-box h3 { font-size: 24px; }
        .sidebar-header { font-size: 20px; }
        .sidebar-nav-item { font-size: 14px; }
        .about-dataset h3 { font-size: 20px; }
        .project-goal h3 { font-size: 20px; }
        .filter-container h4 { font-size: 16px; }
        .contact-links a { font-size: 20px; }
        .mission-section h3 { font-size: 20px; }
        .preview-section h3 { font-size: 20px; }
    }
    </style>
""", unsafe_allow_html=True)

# Expected columns
EXPECTED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'Churn'
]

# Load data with validation
@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("customer_churn_data.csv")
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}. Please ensure the dataset contains all required columns.")
            return None
        categorical_cols = [
            col for col in df.columns 
            if col in EXPECTED_COLUMNS and col != 'customerID' and df[col].dtype == 'object'
        ]
        for col in categorical_cols:
            if df[col].isnull().any():
                st.warning(f"Column {col} contains missing values. Filling with most frequent value.")
                df[col] = df[col].fillna(df[col].mode()[0])
        return df
    except FileNotFoundError:
        st.error("Dataset file 'customer_churn_data.csv' not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}. Please check the file format and content.")
        return None

# Preprocess data
def preprocess_data(df):
    try:
        df_clean = df.copy()
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        if df_clean['TotalCharges'].isnull().any():
            st.warning("Found missing or invalid values in TotalCharges. Imputing with median.")
            df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())
        le_dict = {}
        categorical_cols = [
            col for col in df_clean.columns 
            if col in EXPECTED_COLUMNS and col != 'customerID' and df_clean[col].dtype == 'object'
        ]
        for col in categorical_cols:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            le_dict[col] = le
        return df_clean, le_dict
    except Exception as e:
        st.error(f"Error in preprocessing: {e}. Please check data consistency.")
        return None, None

# Train model
def train_model(X, y, model_type='RandomForest', n_estimators=100, max_depth=None):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        if model_type == 'RandomForest':
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, X_test, y_test, y_pred
    except Exception as e:
        st.error(f"Error training model: {e}. Please check feature data.")
        return None, None, None, None

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'model_params' not in st.session_state:
    st.session_state.model_params = {'n_estimators': 100, 'max_depth': None}
if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'quick_filter' not in st.session_state:
    st.session_state.quick_filter = None

# Sidebar Navigation
st.sidebar.markdown("""
    <div>
        <img src="https://img.icons8.com/color/100/000000/bar-chart.png" class="sidebar-logo">
        <h2 class="sidebar-header">Churn Analytics</h2>
        <p class="sidebar-subtext">Analyze customer behavior and predict churn.</p>
    </div>
""", unsafe_allow_html=True)

# Navigation options
nav_options = [
    ("Home", "Home"),
    ("Dashboard", "Dashboard"),
    ("Data Overview", "Data Overview"),
    ("Exploratory Data Analysis", "EDA"),
    ("Churn Prediction", "Churn Prediction")
]

st.sidebar.markdown("### Navigate")
page = st.sidebar.selectbox(
    "Choose a page",
    options=[opt[1] for opt in nav_options],
    format_func=lambda x: next(opt[0] for opt in nav_options if opt[1] == x),
    index=[opt[1] for opt in nav_options].index(st.session_state.page),
    key="nav_select"
)
if page != st.session_state.page:
    st.session_state.page = page
    st.rerun()

# Quick Filters
with st.sidebar.expander("Quick Filters", expanded=False):
    st.markdown("<h4>Apply Quick Filters</h4>", unsafe_allow_html=True)
    if st.button("High-Risk Customers", key="high_risk"):
        st.session_state.quick_filter = {
            'Contract': ['Month-to-month'],
            'tenure': (0, 12),
            'MonthlyCharges': (70, float('inf'))
        }
        st.rerun()
    if st.button("Senior Citizens", key="senior"):
        st.session_state.quick_filter = {'SeniorCitizen': [1]}
        st.rerun()
    if st.button("Reset Filters", key="reset_filters"):
        st.session_state.quick_filter = None
        st.rerun()

# Model Parameters
with st.sidebar.expander("Model Parameters", expanded=False):
    st.markdown("<h4>Adjust Model Settings</h4>", unsafe_allow_html=True)
    model_type = st.selectbox("Model Type", ["RandomForest", "LogisticRegression"], key="sidebar_model_type")
    if model_type == "RandomForest":
        n_estimators = st.slider("Number of Trees", 50, 200, st.session_state.model_params.get('n_estimators', 100), step=10, key="n_estimators")
        max_depth = st.slider("Max Depth", 5, 50, st.session_state.model_params.get('max_depth', None) or 10, step=5, key="max_depth")
        st.session_state.model_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    else:
        st.session_state.model_params = {}

# Download Reports
with st.sidebar.expander("Download Reports", expanded=False):
    st.markdown("<h4>Export Data & Reports</h4>", unsafe_allow_html=True)
    if st.session_state.df is not None:
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name="filtered_churn_data.csv",
            mime="text/csv",
            key="download_data"
        )
        st.download_button(
            label="Download Analysis Report",
            data="Customer Churn Analysis Report\nGenerated by Churn Analytics Dashboard\n\nKey Metrics and Insights\n[Placeholder for detailed report content]",
            file_name="churn_analysis_report.txt",
            mime="text/plain",
            key="download_report"
        )

# Contact Us
with st.sidebar.expander("Contact Us", expanded=True):
    st.markdown("<h4>Connect with Us</h4>", unsafe_allow_html=True)
    st.markdown("""
        <div class="contact-links">
            <a href="mailto:support@churnanalytics.com" target="_blank" title="Email Support"><i class="fas fa-envelope"></i></a>
            <a href="https://x.com/xai" target="_blank" title="Follow on X"><i class="fab fa-x-twitter"></i></a>
            <a href="https://www.linkedin.com/company/xai" target="_blank" title="Connect on LinkedIn"><i class="fab fa-linkedin"></i></a>
        </div>
    """, unsafe_allow_html=True)

# Help & Support
with st.sidebar.expander("Help & Support", expanded=False):
    st.markdown("""
        <div class="collapsible-section">
            <h4>Help Center</h4>
            <p>Find answers to common questions and get support.</p>
            <ul style="list-style: none; padding: 0;">
                <li style="margin: 10px 0;"><b>How to use the dashboard?</b><br>Navigate through the sections using the sidebar menu. Use filters to customize views and explore predictions.</li>
                <li style="margin: 10px 0;"><b>What data is required?</b><br>The dataset must include columns: customerID, gender, SeniorCitizen, etc. Check the Data Overview for details.</li>
                <li style="margin: 10px 0;"><b>Need help?</b><br>Contact us via the links above.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Sidebar Footer
st.sidebar.markdown("""
    <div class="sidebar-footer">
        <p>Churn Analytics Dashboard v1.0.0</p>
        <p>Developed by xAI | <a href="https://x.ai" style="color: #2DD4BF;">Learn more about xAI</a></p>
    </div>
""", unsafe_allow_html=True)

# Main Content
df = st.session_state.df
if df is not None:
    # Apply quick filters if set
    filtered_df = df.copy()
    if st.session_state.quick_filter:
        for key, value in st.session_state.quick_filter.items():
            try:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                elif isinstance(value, tuple):
                    filtered_df = filtered_df[filtered_df[key].between(value[0], value[1])]
                elif isinstance(value, (int, float)):
                    filtered_df = filtered_df[filtered_df[key] == value]
            except Exception as e:
                st.error(f"Error applying filter on {key}: {e}")

    # Home Page
    if st.session_state.page == "Home":
        st.markdown("""
            <div class="hero-section">
                <h1 class="main-header">Customer Churn Analysis Dashboard</h1>
                <p class="hero-text">
                    A comprehensive platform for analyzing customer behavior, identifying churn drivers, and predicting at-risk customers to inform strategic retention efforts.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Mission Statement
        st.markdown('<p class="sub-header">Our Mission</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="mission-section">
                <h3>Empowering Data-Driven Retention</h3>
                <p>
                    The Customer Churn Analysis Dashboard is designed to provide businesses with actionable insights to reduce customer churn. 
                    By leveraging advanced analytics and predictive modeling, our platform enables organizations to understand customer behavior, 
                    identify key risk factors, and implement effective retention strategies to foster long-term customer loyalty.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Key Metrics Section
        st.markdown('<p class="sub-header">Key Metrics</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        churn_rate = filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
        total_customers = len(filtered_df)
        avg_tenure = filtered_df['tenure'].mean()
        avg_monthly = filtered_df['MonthlyCharges'].mean()

        with col1:
            st.markdown(f"""
                <div class="metric-box churn">
                    <h3>{churn_rate:.1f}%</h3>
                    <p>Churn Rate</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-box customers">
                    <h3>{total_customers:,}</h3>
                    <p>Total Customers</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-box tenure">
                    <h3>{avg_tenure:.1f}</h3>
                    <p>Average Tenure (Months)</p>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="metric-box charges">
                    <h3>${avg_monthly:.2f}</h3>
                    <p>Average Monthly Charges</p>
                </div>
            """, unsafe_allow_html=True)

        # Data Preview
        st.markdown('<p class="sub-header">Dataset Preview</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="preview-section">
                <h3>Sample Customer Data</h3>
                <p>View a snapshot of the customer dataset to understand the data structure.</p>
            </div>
        """, unsafe_allow_html=True)
        st.dataframe(filtered_df.head(5), use_container_width=True)

        # Call to Action
        st.markdown('<p class="sub-header">Get Started</p>', unsafe_allow_html=True)
        if st.button("Explore the Dashboard", key="explore_dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()

        # About Dataset Section
        st.markdown('<p class="sub-header">About the Dataset</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="about-dataset">
                <h3>Dataset Overview</h3>
                <p>The telecom customer dataset provides comprehensive data to support churn analysis and retention strategies. Key features include:</p>
                <ul>
                    <li>Customer Demographics: Gender, Senior Citizen status, Partner, Dependents.</li>
                    <li>Service Subscriptions: Phone, Internet, Online Security, Streaming, and more.</li>
                    <li>Billing Information: Contract types, Payment methods, Monthly and Total Charges.</li>
                    <li>Churn Status: Indicates whether a customer has churned (Yes/No).</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Footer Section
        st.markdown("""
            <div class="footer-section">
                <p>Churn Analytics Dashboard v1.0.0</p>
                <p>Powered by <a href="https://x.ai" target="_blank">xAI</a> | <a href="mailto:support@churnanalytics.com">Contact Support</a> | <a href="https://x.ai/grok">Learn More</a></p>
            </div>
        """, unsafe_allow_html=True)

    # Dashboard Page
    elif st.session_state.page == "Dashboard":
        st.markdown("""
            <div style="background: linear-gradient(90deg, #1E3A8A, #2DD4BF); padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); margin-bottom: 20px;">
                <h1 style="color: #FFFFFF; font-size: 36px; font-weight: 700; text-align: center; margin: 0;">
                    Churn Analytics Dashboard
                </h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Filters
        st.markdown('<div class="filter-container"><h4>Filter Data</h4></div>', unsafe_allow_html=True)
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_gender = st.multiselect(
                    "Gender",
                    options=df['gender'].unique(),
                    default=df['gender'].unique(),
                    key="dash_gender"
                )
            with col2:
                selected_contract = st.multiselect(
                    "Contract",
                    options=df['Contract'].unique(),
                    default=df['Contract'].unique(),
                    key="dash_contract"
                )
            with col3:
                tenure_range = st.slider(
                    "Tenure Range (months)",
                    int(df['tenure'].min()),
                    int(df['tenure'].max()),
                    (int(df['tenure'].min()), int(df['tenure'].max())),
                    key="dash_tenure"
                )
            filtered_df = filtered_df[
                (filtered_df['gender'].isin(selected_gender)) &
                (filtered_df['Contract'].isin(selected_contract)) &
                (filtered_df['tenure'].between(tenure_range[0], tenure_range[1]))
            ]
        
        # Key Metrics
        st.markdown('<p class="sub-header">Key Metrics</p>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        churn_rate = filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
        total_customers = len(filtered_df)
        avg_tenure = filtered_df['tenure'].mean()
        avg_monthly = filtered_df['MonthlyCharges'].mean()
        senior_pct = (filtered_df['SeniorCitizen'] == 1).mean() * 100

        with col1:
            st.markdown(f"""
                <div class="metric-box churn">
                    <h3>{churn_rate:.1f}%</h3>
                    <p>Churn Rate</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-box customers">
                    <h3>{total_customers:,}</h3>
                    <p>Total Customers</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-box tenure">
                    <h3>{avg_tenure:.1f}</h3>
                    <p>Average Tenure (Months)</p>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="metric-box charges">
                    <h3>${avg_monthly:.2f}</h3>
                    <p>Average Monthly Charges</p>
                </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
                <div class="metric-box senior">
                    <h3>{senior_pct:.1f}%</h3>
                    <p>Senior Citizens</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown('<p class="sub-header">Visual Insights</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn Distribution
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig1 = px.pie(
                filtered_df,
                names='Churn',
                title="Churn Distribution",
                color_discrete_sequence=['#F43F5E', '#2DD4BF'],
                template='plotly_dark'
            )
            fig1.update_layout(title_x=0.5, margin=dict(t=50, b=20))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Tenure vs Monthly Charges
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig2 = px.scatter(
                filtered_df,
                x='tenure',
                y='MonthlyCharges',
                color='Churn',
                title="Tenure vs Monthly Charges",
                color_discrete_sequence=['#F43F5E', '#2DD4BF'],
                hover_data=['Contract'],
                template='plotly_dark'
            )
            fig2.update_layout(title_x=0.5, margin=dict(t=50, b=20))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Contract Type vs Churn
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig3 = px.histogram(
                filtered_df,
                x='Contract',
                color='Churn',
                barmode='group',
                title="Contract Type vs Churn",
                color_discrete_sequence=['#F43F5E', '#2DD4BF'],
                template='plotly_dark'
            )
            fig3.update_layout(title_x=0.5, margin=dict(t=50, b=20))
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Internet Service vs Churn
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig4 = px.histogram(
                filtered_df,
                x='InternetService',
                color='Churn',
                barmode='group',
                title="Internet Service vs Churn",
                color_discrete_sequence=['#F43F5E', '#2DD4BF'],
                template='plotly_dark'
            )
            fig4.update_layout(title_x=0.5, margin=dict(t=50, b=20))
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Data Overview
    elif st.session_state.page == "Data Overview":
        st.markdown('<p class="main-header">Data Overview</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.write("### Dataset Preview")
        st.dataframe(filtered_df.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.write("### Column Descriptions")
        st.markdown("""
        - **customerID**: Unique identifier for each customer.
        - **gender**: Gender of the customer (Male, Female).
        - **SeniorCitizen**: Whether the customer is a senior citizen (1: Yes, 0: No).
        - **Partner**: Whether the customer has a partner (Yes, No).
        - **Dependents**: Whether the customer has dependents (Yes, No).
        - **tenure**: Number of months the customer has stayed with the company.
        - **PhoneService**: Whether the customer has phone service (Yes, No).
        - **MultipleLines**: Whether the customer has multiple lines (Yes, No, No phone service).
        - **InternetService**: Customer’s internet service provider (DSL, Fiber optic, No).
        - **OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies**: Additional services (Yes, No, No internet service).
        - **Contract**: Contract term (Month-to-month, One year, Two year).
        - **PaperlessBilling**: Whether the customer uses paperless billing (Yes, No).
        - **PaymentMethod**: Payment method (Electronic check, Mailed check, Bank transfer, Credit card).
        - **MonthlyCharges**: Monthly charges incurred by the customer.
        - **TotalCharges**: Total charges incurred by the customer.
        - **Churn**: Whether the customer churned (Yes, No).
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.write("### Basic Statistics")
        st.dataframe(filtered_df.describe(include='all'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.write("### Missing Values")
        missing = filtered_df.isnull().sum()
        st.write(missing[missing > 0] if missing.sum() > 0 else "No missing values.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Exploratory Data Analysis
    elif st.session_state.page == "EDA":
        st.markdown('<p class="main-header">Exploratory Data Analysis</p>', unsafe_allow_html=True)
        
        # Data Filtering
        st.markdown('<div class="filter-container"><h4>Filter Data</h4></div>', unsafe_allow_html=True)
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_gender = st.multiselect(
                    "Gender",
                    options=df['gender'].unique(),
                    default=df['gender'].unique(),
                    key="eda_gender"
                )
            with col2:
                selected_contract = st.multiselect(
                    "Contract",
                    options=df['Contract'].unique(),
                    default=df['Contract'].unique(),
                    key="eda_contract"
                )
            with col3:
                tenure_range = st.slider(
                    "Tenure Range (months)",
                    int(df['tenure'].min()),
                    int(df['tenure'].max()),
                    (int(df['tenure'].min()), int(df['tenure'].max())),
                    key="eda_tenure"
                )
            filtered_df = filtered_df[
                (filtered_df['gender'].isin(selected_gender)) &
                (filtered_df['Contract'].isin(selected_contract)) &
                (filtered_df['tenure'].between(tenure_range[0], tenure_range[1]))
            ]
        
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.write(f"Filtered dataset size: {filtered_df.shape[0]:,} rows")
        st.markdown('</div>', unsafe_allow_html=True)

        # Churn Distribution
        st.markdown('<p class="sub-header">Churn Distribution</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig = px.histogram(
            filtered_df,
            x='Churn',
            color='Churn',
            title="Churn Distribution",
            color_discrete_sequence=['#F43F5E', '#2DD4BF'],
            template='plotly_dark'
        )
        fig.update_layout(title_x=0.5, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Numerical Features
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        st.markdown('<p class="sub-header">Numerical Features Analysis</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        num_feature = st.selectbox(
            "Select Numerical Feature",
            numerical_cols,
            key="num_feature"
        )
        fig = px.histogram(
            filtered_df,
            x=num_feature,
            color='Churn',
            marginal="box",
            title=f"{num_feature} Distribution by Churn",
            color_discrete_sequence=['#F43F5E', '#2DD4BF'],
            template='plotly_dark'
        )
        fig.update_layout(title_x=0.5, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Categorical Features
        categorical_cols = [
            col for col in df.columns 
            if col in EXPECTED_COLUMNS and col != 'customerID' and df[col].dtype == 'object'
        ]
        st.markdown('<p class="sub-header">Categorical Features Analysis</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        cat_feature = st.selectbox(
            "Select Categorical Feature",
            categorical_cols,
            key="cat_feature"
        )
        fig = px.histogram(
            filtered_df,
            x=cat_feature,
            color='Churn',
            barmode='group',
            title=f"{cat_feature} vs Churn",
            color_discrete_sequence=['#F43F5E', '#2DD4BF'],
            template='plotly_dark'
        )
        fig.update_layout(title_x=0.5, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Correlation Matrix
        st.markdown('<p class="sub-header">Correlation Matrix (Numerical Features)</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        numeric_df = filtered_df[numerical_cols].copy()
        numeric_df['TotalCharges'] = pd.to_numeric(numeric_df['TotalCharges'], errors='coerce')
        numeric_df = numeric_df.fillna(numeric_df.median())
        corr = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}"
        ))
        fig.update_layout(title="Correlation Matrix", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Churn Prediction
    elif st.session_state.page == "Churn Prediction":
        st.markdown('<p class="main-header">Churn Prediction Model</p>', unsafe_allow_html=True)
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            df_clean, le_dict = preprocess_data(df)
        
        if df_clean is not None:
            # Features and target
            X = df_clean.drop(['customerID', 'Churn'], axis=1)
            y = df_clean['Churn']
            
            # Model selection
            st.markdown('<div class="filter-container">', unsafe_allow_html=True)
            model_type = st.selectbox(
                "Select Model",
                ["RandomForest", "LogisticRegression"],
                help="Choose the machine learning model for prediction.",
                key="model_type_main",
                index=["RandomForest", "LogisticRegression"].index(st.session_state.get('sidebar_model_type', 'RandomForest'))
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Train model
            with st.spinner("Training model..."):
                model_params = st.session_state.get('model_params', {})
                model, X_test, y_test, y_pred = train_model(
                    X, y, model_type,
                    n_estimators=model_params.get('n_estimators', 100),
                    max_depth=model_params.get('max_depth', None)
                )
            
            if model is not None:
                # Model Performance
                st.markdown('<p class="sub-header">Model Performance</p>', unsafe_allow_html=True)
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.2f}")
                st.write("**Classification Report**:")
                st.text(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Confusion Matrix
                st.markdown('<p class="sub-header">Confusion Matrix</p>', unsafe_allow_html=True)
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                cm = confusion_matrix(y_test, y_pred)
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    text=cm,
                    texttemplate="%{text}",
                    colorscale='Blues'
                ))
                fig.update_layout(title="Confusion Matrix", template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Feature Importance (only for RandomForest)
                if model_type == 'RandomForest':
                    st.markdown('<p class="sub-header">Feature Importance</p>', unsafe_allow_html=True)
                    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values(by='importance', ascending=False)
                    fig = px.bar(
                        feature_importance,
                        x='importance',
                        y='feature',
                        title="Feature Importance",
                        color_discrete_sequence=['#2DD4BF'],
                        template='plotly_dark'
                    )
                    fig.update_layout(title_x=0.5, margin=dict(t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Predict for a customer
                st.markdown('<p class="sub-header">Predict Churn for a Single Customer</p>', unsafe_allow_html=True)
                st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                st.write("Enter customer details below to predict churn probability.")
                
                with st.form("prediction_form"):
                    cols = st.columns(4)
                    inputs = {}
                    for i, col in enumerate(X.columns):
                        with cols[i % 4]:
                            if col in le_dict:
                                unique_vals = list(df[col].unique())
                                inputs[col] = st.selectbox(
                                    f"{col}",
                                    unique_vals,
                                    help=f"Select a value for {col}",
                                    key=f"pred_{col}"
                                )
                            else:
                                min_val = float(df[col].min())
                                max_val = float(df[col].max())
                                avg_val = float(df[col].mean())
                                inputs[col] = st.number_input(
                                    f"{col}",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=avg_val,
                                    help=f"Enter a value for {col} (range: {min_val:.2f} to {max_val:.2f})",
                                    key=f"input_{col}"
                                )
                    
                    submit = st.form_submit_button("Predict Churn")
                    
                    if submit:
                        with st.spinner("Making prediction..."):
                            input_data = pd.DataFrame([inputs])
                            try:
                                for col in le_dict:
                                    if col in input_data.columns:
                                        if input_data[col].iloc[0] in le_dict[col].classes_:
                                            input_data[col] = le_dict[col].transform([input_data[col].iloc[0]])[0]
                                        else:
                                            st.error(f"Value '{input_data[col].iloc[0]}' in {col} is not recognized. Please select a valid option.")
                                            st.stop()
                                prediction = model.predict(input_data)
                                prob = model.predict_proba(input_data)[0]
                                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                                st.write(f"**Prediction**: {'Churn' if prediction[0] == 1 else 'No Churn'}")
                                st.write(f"**Churn Probability**: {prob[1]:.2%}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Prediction error: {e}. Please check input data.")
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.error("Error: Please ensure the dataset file 'customer_churn_data.csv' is available in the correct directory. Some features will be disabled.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display limited Home page without data
    if st.session_state.page == "Home":
        st.markdown("""
            <div class="hero-section">
                <h1 class="main-header">Customer Churn Analysis Dashboard</h1>
                <p class="hero-text">
                    A comprehensive platform for analyzing customer behavior, identifying churn drivers, and predicting at-risk customers to inform strategic retention efforts.
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="sub-header">Get Started</p>', unsafe_allow_html=True)
        if st.button("Explore the Dashboard", key="explore_dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()

        st.markdown('<p class="sub-header">About the Dataset</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="about-dataset">
                <h3>Dataset Overview</h3>
                <p>The telecom customer dataset provides comprehensive data to support churn analysis and retention strategies. Key features include:</p>
                <ul>
                    <li>Customer Demographics: Gender, Senior Citizen status, Partner, Dependents.</li>
                    <li>Service Subscriptions: Phone, Internet, Online Security, Streaming, and more.</li>
                    <li>Billing Information: Contract types, Payment methods, Monthly and Total Charges.</li>
                    <li>Churn Status: Indicates whether a customer has churned (Yes/No).</li>
                </ul>
                <p><b>Note:</b> Please upload the dataset to enable full functionality.</p>
            </div>
        """, unsafe_allow_html=True)