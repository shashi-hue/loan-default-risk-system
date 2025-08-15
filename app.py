import sys
import os
from pathlib import Path
import gdown

# Get the repository root directory (go up from dashboard/ to repo root)
repo_root = Path(__file__).parent.parent
os.chdir(repo_root)  # Change working directory to repo root
sys.path.insert(0, str(repo_root))  # Add repo root to Python path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
import os
try:
    from src.explainability import LoanExplainer, load_test_data
except Exception as e:
    tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    st.error(f"Startup import error:\n```\n{tb_str}\n```")
    st.stop()

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk system",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .approved {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
    }
    
    .declined {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    """Load model and sample data with caching"""
    try:
        with st.spinner("Loading model and initializing SHAP explainer..."):
            explainer = LoanExplainer()
        
        # Path to test data
        test_path = "data/processed/model_df.parquet"
        
        # Google Drive file ID (same as in explainability.py)
        file_id = "1xypiRV2aUU2jXEUG2k8yqsBwToyVFCqP"
        
        # Download if missing
        #if not os.path.exists(test_path):
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        st.info("Downloading test data from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", test_path, quiet=False)
        
        # Load data
        X_sample, y_sample = load_test_data(test_path)
        
        # Take a reasonable sample for dashboard performance
        sample_size = min(5000, len(X_sample))
        
        # Sample both X and y together to maintain alignment
        sample_indices = X_sample.sample(sample_size, random_state=42).index
        X_sample = X_sample.loc[sample_indices].reset_index(drop=True)
        if y_sample is not None:
            y_sample = y_sample.loc[sample_indices].reset_index(drop=True)
            
        return explainer, X_sample, y_sample
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure model files are available in the 'models/' directory")
        return None, None, None


def create_kpi_metrics(explainer, X_data):
    """Create KPI metrics for executive dashboard"""
    risk_dist = explainer.get_risk_distribution(X_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Approval Rate</h3>
            <h2>{risk_dist['approval_rate']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Avg Risk Score</h3>
            <h2>{risk_dist['mean_risk']:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>High Risk Rate</h3>
            <h2>{risk_dist['high_risk_rate']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Current Threshold</h3>
            <h2>{explainer.threshold:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    return risk_dist


def show_individual_assessment(explainer, X_sample):
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Applicant Information")
        
        # BASIC APPLICANT INFO
        st.markdown("**Personal & Employment Details**")
        
        annual_income = st.number_input(
            "Annual Income ($)",
            value=50000,
            step=1000,
            help="Applicant's gross annual income"
        )
        
        employment_length = st.selectbox(
            "Employment Length",
            options=[
                "Unknown", "< 1 year", "1 year", "2 years", "3 years", "4 years", 
                "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
            ],
            index=6,  # Default to 5 years
            help="Length of current employment"
        )
        
        home_ownership = st.selectbox(
            "Home Ownership",
            options=["Unknown", "RENT", "OWN", "MORTGAGE", "OTHER"],
            index=1,
            help="Housing situation"
        )
        
        verification_status = st.selectbox(
            "Income Verification",
            options=["Unknown", "Not Verified", "Source Verified", "Verified"],
            index=2,
            help="Level of income verification completed"
        )
        
        # CREDIT PROFILE
        st.markdown("**Credit Profile**")
        
        fico_score = st.number_input(
            "FICO Credit Score",
            value=650,
            step=5,
            help="Applicant's FICO credit score (300-850 typical range)"
        )
        
        # SUB GRADE INPUT - ADDED HERE
        st.markdown("**Credit Sub-Grade**")
        sub_grade = create_sub_grade_input()
        
        debt_to_income = st.number_input(
            "Debt-to-Income Ratio (%)",
            value=15.0,
            step=0.5,
            help="Monthly debt payments / Monthly income"
        )
        if st.checkbox("DTI Unknown", key="dti_unknown"):
            debt_to_income = None
        
        credit_utilization = st.number_input(
            "Credit Card Utilization (%)",
            value=30.0,
            step=1.0,
            help="Credit card balance / Credit limit"
        )
        if st.checkbox("Credit Utilization Unknown", key="util_unknown"):
            credit_utilization = None
        
        # CREDIT HISTORY
        st.markdown("**Credit History**")
        
        total_accounts = st.number_input(
            "Total Credit Accounts",
            value=10,
            step=1,
            help="Total number of credit accounts"
        )
        if st.checkbox("Total Accounts Unknown", key="total_acc_unknown"):
            total_accounts = None
        
        open_accounts = st.number_input(
            "Open Credit Accounts",
            value=5,
            step=1,
            help="Number of currently open accounts"
        )
        if st.checkbox("Open Accounts Unknown", key="open_acc_unknown"):
            open_accounts = None
        
        delinquencies_2yrs = st.number_input(
            "Delinquencies (Last 2 Years)",
            value=0,
            step=1,
            help="Number of 30+ day delinquencies in past 2 years"
        )
        if st.checkbox("Delinquencies Unknown", key="delinq_unknown"):
            delinquencies_2yrs = None
        
        credit_inquiries = st.number_input(
            "Credit Inquiries (Last 6 Months)",
            value=1,
            step=1,
            help="Number of hard credit inquiries"
        )
        if st.checkbox("Credit Inquiries Unknown", key="inq_unknown"):
            credit_inquiries = None
        
        # LOAN DETAILS
        st.markdown("**Loan Details**")
        
        loan_amount = st.number_input(
            "Requested Loan Amount ($)",
            value=15000,
            step=500,
            help="Amount of loan requested"
        )
        
        loan_purpose = st.selectbox(
            "Loan Purpose",
            options=[
                "debt_consolidation", "credit_card", "home_improvement", 
                "major_purchase", "medical", "car", "vacation", 
                "wedding", "moving", "house", "other"
            ],
            index=0,
            help="Primary purpose of the loan"
        )
        
        loan_term = st.selectbox(
            "Loan Term",
            options=["36 months", "60 months"],
            index=0,
            help="Length of loan repayment period"
        )
        
        interest_rate = st.number_input(
            "Interest Rate (%)",
            value=12.0,
            step=0.25,
            help="Proposed interest rate"
        )
    
    with col2:
        st.subheader("📊 Risk Assessment Results")
        
        if st.button("🔍 Analyze Loan Application", type="primary", use_container_width=True):
            try:
                # Convert user inputs to model features
                model_input = convert_user_inputs_to_model_features_flexible(
                    annual_income=annual_income,
                    employment_length=employment_length,
                    home_ownership=home_ownership,
                    verification_status=verification_status,
                    fico_score=fico_score,
                    sub_grade=sub_grade,  # ADDED SUB_GRADE PARAMETER
                    debt_to_income=debt_to_income,
                    credit_utilization=credit_utilization,
                    total_accounts=total_accounts,
                    open_accounts=open_accounts,
                    delinquencies_2yrs=delinquencies_2yrs,
                    credit_inquiries=credit_inquiries,
                    loan_amount=loan_amount,
                    loan_purpose=loan_purpose,
                    loan_term=loan_term,
                    interest_rate=interest_rate,
                    X_sample=X_sample
                )
                
                # Get prediction
                pred_info = explainer.predict_loan(model_input)
                
                # Display results
                decision_color = "🟢" if pred_info['decision'] == 'Approved' else "🔴"
                risk_level = get_risk_level(pred_info['calibrated_probability'])
                
                # Main result
                st.markdown(f"""
                ## {decision_color} **{pred_info['decision']}**
                
                ### 📈 **Risk Assessment:**
                - **Default Probability:** {pred_info['calibrated_probability']:.1%}
                - **Risk Level:** {risk_level}
                - **Decision Confidence:** {pred_info.get('confidence', 0.85):.1%}
                """)
                
                # Risk breakdown
                col_risk1, col_risk2 = st.columns(2)
                
                with col_risk1:
                    # Risk gauge
                    fig_gauge = create_risk_gauge(pred_info['calibrated_probability'], pred_info['threshold'])
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col_risk2:
                    # Key factors (only show if values were provided)
                    sub_grade_letter = convert_sub_grade_to_letter(sub_grade)
                    factors_text = "**Key Decision Factors:**\n"
                    factors_text += f"- FICO Score: {fico_score} {'✅' if fico_score >= 650 else '❌'}\n"
                    factors_text += f"- Credit Sub-Grade: {sub_grade_letter} {'✅' if sub_grade <= 3.0 else '❌'}\n"
                    if debt_to_income is not None:
                        factors_text += f"- Debt-to-Income: {debt_to_income:.1f}% {'✅' if debt_to_income <= 20 else '❌'}\n"
                    if credit_utilization is not None:
                        factors_text += f"- Credit Utilization: {credit_utilization:.0f}% {'✅' if credit_utilization <= 30 else '❌'}\n"
                    if employment_length != "Unknown":
                        factors_text += f"- Employment: {employment_length} {'✅' if employment_length not in ['< 1 year', 'Unknown'] else '❌'}\n"
                    
                    st.markdown(factors_text)
                
                # SHAP explanation
                with st.expander("📊 View Detailed Risk Analysis"):
                    try:
                        fig_shap, _, _ = explainer.explain_single_loan(model_input)
                        st.pyplot(fig_shap, clear_figure=True)
                    except Exception as shap_error:
                        st.warning(f"Could not generate SHAP explanation: {str(shap_error)}")
                
            except Exception as e:
                st.error(f"Error in risk assessment: {str(e)}")
                with st.expander("Debug Information"):
                    st.write(f"Error details: {str(e)}")

# NEW HELPER FUNCTIONS FOR SUB_GRADE

def create_sub_grade_input():
    """Create sub-grade input slider with A1-G5 options"""
    # Generate all sub-grade options (A1-G5 mapped to 1.1-7.5)
    grade_options = []
    grade_values = []
    grade_labels = []
    
    for grade_idx, grade_letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 1):
        for sub_idx in range(1, 6):
            grade_value = float(f"{grade_idx}.{sub_idx}")
            grade_label = f"{grade_letter}{sub_idx}"
            grade_display = f"{grade_label} ({grade_value})"
            
            grade_options.append(grade_display)
            grade_values.append(grade_value)
            grade_labels.append(grade_label)
    
    # Create selectbox
    selected_display = st.selectbox(
        "Credit Sub-Grade",
        options=grade_options,
        index=10,  # Default to C1 (3.1) - middle grade
        help="Credit sub-grade assigned by lender (A1=Best, G5=Worst)"
    )
    
    # Extract the numeric value
    selected_index = grade_options.index(selected_display)
    return grade_values[selected_index]

def convert_sub_grade_to_letter(sub_grade_value):
    """Convert numeric sub_grade back to letter format (e.g., 1.1 -> A1)"""
    try:
        grade_part = int(sub_grade_value)
        sub_part = int((sub_grade_value - grade_part) * 10)
        grade_letter = chr(64 + grade_part)  # A=65, B=66, etc.
        return f"{grade_letter}{sub_part}"
    except:
        return f"Grade {sub_grade_value:.1f}"

# UPDATED CONVERSION FUNCTION

def convert_user_inputs_to_model_features_flexible(annual_income, employment_length, home_ownership, 
                                                 verification_status, fico_score, sub_grade,  # ADDED SUB_GRADE
                                                 debt_to_income, credit_utilization, total_accounts, 
                                                 open_accounts, delinquencies_2yrs, credit_inquiries, 
                                                 loan_amount, loan_purpose, loan_term, interest_rate, X_sample):
    """Convert user-friendly inputs to model features with null handling"""
    
    # Create base DataFrame with model structure
    input_df = pd.DataFrame(index=[0])
    
    # Fill with NaN first (XGBoost handles missing values)
    for col in X_sample.columns:
        input_df[col] = np.nan
    
    # Transform user inputs to model features (only if provided)
    
    # Handle annual income - always provided now
    input_df['annual_inc_log'] = np.log1p(annual_income)
    
    # Handle debt-to-income
    if debt_to_income is not None:
        input_df['dti_log'] = np.log1p(debt_to_income)
    
    # Direct mappings - fico_score and sub_grade always provided now
    input_df['fico_score'] = fico_score
    input_df['sub_grade'] = sub_grade  # ADDED SUB_GRADE MAPPING
    
    if credit_utilization is not None:
        input_df['revol_util'] = credit_utilization
    if total_accounts is not None:
        input_df['total_acc'] = total_accounts
    if open_accounts is not None:
        input_df['open_acc'] = open_accounts
    if delinquencies_2yrs is not None:
        input_df['delinq_2yrs'] = delinquencies_2yrs
    if credit_inquiries is not None:
        input_df['inq_last_6mths'] = credit_inquiries
    
    # Loan details (always provided)
    input_df['loan_amnt'] = loan_amount
    input_df['int_rate'] = interest_rate
    
    # Employment length mapping
    if employment_length != "Unknown":
        emp_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8,
            '9 years': 9, '10+ years': 10
        }
        input_df['emp_length'] = emp_map.get(employment_length, np.nan)
    
    # Term mapping
    input_df['term'] = 0 if loan_term == '36 months' else 1
    
    # Verification status
    if verification_status != "Unknown":
        input_df['verification_status_Verified'] = 1 if verification_status == 'Verified' else 0
    
    # Home ownership (only set if provided)
    if home_ownership != "Unknown":
        home_cols = [col for col in X_sample.columns if col.startswith('home_ownership_')]
        for col in home_cols:
            input_df[col] = 0
        
        if f'home_ownership_{home_ownership}' in input_df.columns:
            input_df[f'home_ownership_{home_ownership}'] = 1
    
    # Purpose
    purpose_cols = [col for col in X_sample.columns if col.startswith('purpose_')]
    for col in purpose_cols:
        input_df[col] = 0
    
    if f'purpose_{loan_purpose}' in input_df.columns:
        input_df[f'purpose_{loan_purpose}'] = 1
    
    # Calculate derived features - annual_income always available now
    monthly_income = annual_income / 12
    installment = calculate_monthly_payment(loan_amount, interest_rate, loan_term)
    input_df['installment'] = installment
    input_df['installment_income_ratio'] = installment / (monthly_income + 1)
    input_df['loan_amount_income_ratio'] = loan_amount / (annual_income + 1)
    
    if debt_to_income is not None:
        input_df['fico_dti_ratio'] = fico_score / (1 + debt_to_income)
    
    # Additional engineered features
    if 'revolutil_fico' in X_sample.columns and credit_utilization is not None:
        input_df['revolutil_fico'] = credit_utilization * fico_score
    
    return input_df

# KEEP ALL OTHER HELPER FUNCTIONS THE SAME
def get_risk_level(probability):
    """Convert probability to risk level using balanced industry thresholds"""
    if probability <= 0.15:
        return "🟢 Low Risk"
    elif probability <= 0.30:
        return "🟡 Medium Risk" 
    elif probability <= 0.50:
        return "🟠 High Risk"
    else:
        return "🔴 Very High Risk"

def create_risk_gauge(probability, threshold):
    """Create risk probability gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default Risk %"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 25], 'color': "yellow"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def calculate_monthly_payment(loan_amount, annual_rate, term_months):
    """Calculate monthly loan payment"""
    if isinstance(term_months, str):
        term_months = 36 if "36" in term_months else 60
    
    monthly_rate = annual_rate / 100 / 12
    if monthly_rate == 0:
        return loan_amount / term_months
    
    return loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)



def show_batch_analysis(explainer, X_sample, y_sample=None):
    """Enhanced batch loan risk analysis with optimized sampling"""
    st.header("📊 Portfolio Batch Risk Analysis")
    
    # Sample size configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Portfolio Risk Assessment:** Analyze risk distribution across your loan portfolio 
        to identify concentration risks, optimal approval rates, and expected loss patterns. 
        This analysis helps in portfolio optimization and risk management strategies.
        """)
    
    with col2:
        # Dynamic sample size based on data availability
        max_samples = len(X_sample)
        default_sample = min(10000, max_samples)  # Increased from 100
        
        sample_size = st.slider(
            "Portfolio Sample Size",
            min_value=1000,
            max_value=min(50000, max_samples),
            value=default_sample,
            step=1000,
            help="Larger samples provide more accurate risk estimates but take longer to process"
        )
    
    if st.button("🚀 Run Risk Analysis", type="primary"):
        with st.spinner(f"Analyzing {sample_size:,} loans..."):
            # Optimized sampling strategy
            if sample_size < len(X_sample):
                # Stratified sampling if labels available
                if y_sample is not None:
                    from sklearn.model_selection import train_test_split
                    X_batch, _, y_batch, _ = train_test_split(
                        X_sample, y_sample, 
                        train_size=sample_size/len(X_sample),
                        stratify=y_sample,
                        random_state=42
                    )
                else:
                    # Random sampling
                    X_batch = X_sample.sample(sample_size, random_state=42)
                    y_batch = None
            else:
                X_batch = X_sample
                y_batch = y_sample
            
            # Batch predictions with progress tracking
            progress_bar = st.progress(0)
            
            # Process in chunks to avoid memory issues
            chunk_size = 5000
            all_results = []
            
            for i in range(0, len(X_batch), chunk_size):
                chunk = X_batch.iloc[i:i+chunk_size]
                chunk_results = explainer.batch_predict(chunk)
                all_results.append(chunk_results)
                
                # Update progress
                progress = min((i + chunk_size) / len(X_batch), 1.0)
                progress_bar.progress(progress)
            
            # Combine results
            batch_results = pd.concat(all_results, ignore_index=True)
            progress_bar.empty()
            
            # Display risk distribution analysis
            show_risk_distribution_analysis(batch_results, X_batch, y_batch)

def show_risk_distribution_analysis(batch_results, X_batch, y_batch):
    """Enhanced risk distribution analysis"""
    st.subheader("📈 Portfolio Risk Distribution")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        approval_rate = (batch_results['decision'] == 'Approved').mean()
        st.metric(
            "Approval Rate", 
            f"{approval_rate:.1%}",
            help="Percentage of applications that would be approved"
        )
    
    with col2:
        avg_risk = batch_results['calibrated_probability'].mean()
        st.metric(
            "Average Risk", 
            f"{avg_risk:.2%}",
            help="Average predicted default probability"
        )
    
    with col3:
        high_risk_rate = (batch_results['calibrated_probability'] > 0.3).mean()
        st.metric(
            "High Risk Rate", 
            f"{high_risk_rate:.1%}",
            help="Percentage of high-risk applications (>30% default probability)"
        )
    
    with col4:
        risk_concentration = batch_results['calibrated_probability'].std()
        st.metric(
            "Risk Concentration", 
            f"{risk_concentration:.3f}",
            help="Standard deviation of risk scores (higher = more diverse risk)"
        )
    
    # Risk distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk histogram
        fig_hist = px.histogram(
            batch_results, 
            x='calibrated_probability',
            nbins=50,
            title="Risk Score Distribution",
            labels={'calibrated_probability': 'Default Probability', 'count': 'Number of Loans'}
        )
        fig_hist.add_vline(
            x=batch_results['calibrated_probability'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text="Average Risk"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Decision distribution
        decision_counts = batch_results['decision'].value_counts()
        fig_pie = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            title="Approval Decision Distribution",
            color_discrete_map={'Approved': 'green', 'Declined': 'red'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Risk by category breakdown
    if 'risk_category' in batch_results.columns:
        st.subheader("📊 Risk Category Breakdown")
        
        risk_summary = batch_results.groupby('risk_category').agg({
            'calibrated_probability': ['count', 'mean', 'std'],
            'decision': lambda x: (x == 'Approved').mean()
        }).round(3)
        
        risk_summary.columns = ['Count', 'Avg_Risk', 'Risk_Std', 'Approval_Rate']
        risk_summary['Percentage'] = (risk_summary['Count'] / len(batch_results) * 100).round(1)
        
        st.dataframe(risk_summary, use_container_width=True)
    
    # Expected loss calculation if actual outcomes available
    if y_batch is not None:
        st.subheader("💰 Expected vs Actual Performance")
        comparison_df = pd.DataFrame({
            'Predicted_Risk': batch_results['calibrated_probability'],
            'Actual_Default': y_batch.values
        })
        
        # Binned analysis
        comparison_df['Risk_Bin'] = pd.cut(
            comparison_df['Predicted_Risk'], 
            bins=10, 
            labels=[f"{i*10}-{(i+1)*10}%" for i in range(10)]
        )
        
        bin_analysis = comparison_df.groupby('Risk_Bin').agg({
            'Predicted_Risk': 'mean',
            'Actual_Default': 'mean'
        }).reset_index()
        
        fig_calibration = px.scatter(
            bin_analysis,
            x='Predicted_Risk',
            y='Actual_Default',
            title="Model Calibration: Predicted vs Actual Default Rates",
            labels={'Predicted_Risk': 'Predicted Default Rate', 'Actual_Default': 'Actual Default Rate'}
        )
        fig_calibration.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dash", color="red"),
            name="Perfect Calibration"
        )
        st.plotly_chart(fig_calibration, use_container_width=True)



def plot_risk_distribution(X_data, explainer):
    """Create risk distribution plot"""
    probabilities = explainer.cal_model.predict_proba(X_data)[:, 1]
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=probabilities,
        nbinsx=50,
        name='Risk Distribution',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # Add threshold line
    fig.add_vline(
        x=explainer.threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {explainer.threshold:.3f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Portfolio Risk Distribution",
        xaxis_title="Default Probability",
        yaxis_title="Number of Loans",
        showlegend=False,
        height=400
    )
    
    return fig



def plot_threshold_analysis(explainer, X_data):
    """Create threshold impact analysis"""
    threshold_df = explainer.simulate_threshold_impact(X_data)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Approval Rate vs Threshold', 'Expected Bad Rate vs Threshold'],
        vertical_spacing=0.15
    )
    
    # Approval rate
    fig.add_trace(
        go.Scatter(x=threshold_df['threshold'], y=threshold_df['approval_rate'],
                  mode='lines+markers', name='Approval Rate', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Bad rate
    fig.add_trace(
        go.Scatter(x=threshold_df['threshold'], y=threshold_df['expected_bad_rate'],
                  mode='lines+markers', name='Expected Bad Rate', line=dict(color='red')),
        row=2, col=1
    )
    
    # Current threshold line
    fig.add_vline(x=explainer.threshold, line_dash="dash", line_color="green",
                 annotation_text="Current Threshold")
    
    fig.update_layout(height=500, title_text="Threshold Impact Analysis")
    fig.update_xaxes(title_text="Threshold", row=2, col=1)
    fig.update_yaxes(title_text="Approval Rate", row=1, col=1)
    fig.update_yaxes(title_text="Bad Rate", row=2, col=1)
    
    return fig


def main():
    st.title("🏦 Loan Default Risk System Dashboard")
    st.markdown("Advanced ML-powered loan risk assessment and portfolio optimization")
    
    # Load model and data
    explainer, X_sample, y_sample = load_model_and_data()
    
    if explainer is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.title("🎛️ Controls")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Executive Dashboard", "🎯 Risk Assessment", "🔍 Explainability", "⚖️ Portfolio Optimizer"]
    )
    
    if page == "📊 Executive Dashboard":
        st.header("Executive Dashboard")
        
        # KPI Metrics
        risk_dist = create_kpi_metrics(explainer, X_sample)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution plot
            risk_fig = plot_risk_distribution(X_sample, explainer)
            st.plotly_chart(risk_fig, use_container_width=True)
            
        with col2:
            # Risk percentiles
            st.subheader("Risk Percentiles")
            perc_df = pd.DataFrame.from_dict(risk_dist['percentiles'], orient='index', columns=['Percentile'])
            perc_df.index.name = 'Level'
            st.dataframe(perc_df, use_container_width=True)
            
            # Portfolio summary
            st.subheader("Portfolio Summary")
            st.metric("Mean Risk", f"{risk_dist['mean_risk']:.3f}")
            st.metric("Risk Std Dev", f"{risk_dist['std_risk']:.3f}")
    
    elif page == "🎯 Risk Assessment":
        st.header("Individual Loan Risk Assessment")
        
        tab1, tab2 = st.tabs(["New Application", "Batch Analysis"])
        
        with tab1:
            show_individual_assessment(explainer, X_sample)
        
        with tab2:
            show_batch_analysis(explainer, X_sample, y_sample)
    
    elif page == "🔍 Explainability":
        st.header("Model Explainability & Feature Analysis")
        
        tab1, tab2 = st.tabs(["Feature Importance", "Feature Interactions"])
        
        with tab1:
            st.subheader("Global Feature Importance")
            
            # Get feature importance
            importance_df = explainer.get_feature_importance(X_sample, top_n=20)
            
            # Plot importance
            fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                        title="Top 20 Most Important Features")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show SHAP summary plot
            st.subheader("SHAP Summary Plot")
            try:
                shap_fig = explainer.create_summary_plot(X_sample.head(500))
                st.pyplot(shap_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating SHAP plot: {str(e)}")
        
        with tab2:
            st.subheader("Feature Dependence Analysis")
            
            # Feature selection
            selected_feature = st.selectbox(
                "Select Feature for Dependence Analysis:",
                options=explainer.feature_names[:20]  # Top 20 for performance
            )
            
            if st.button("Generate Dependence Plot"):
                try:
                    dep_fig = explainer.create_dependence_plot(X_sample.head(500), selected_feature)
                    st.pyplot(dep_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating dependence plot: {str(e)}")
    
    elif page == "⚖️ Portfolio Optimizer":
        st.header("Portfolio Optimization & Threshold Tuning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Settings")
            st.metric("Current Threshold", explainer.threshold)
            
            # Threshold slider (for visualization only)
            new_threshold = st.slider(
                "Explore Threshold Impact:",
                min_value=0.1, max_value=0.5, 
                value=float(explainer.threshold), step=0.01
            )
            
            # Calculate impact of new threshold
            probabilities = explainer.cal_model.predict_proba(X_sample)[:, 1]
            new_approval_rate = (probabilities <= new_threshold).mean()
            approved_loans = probabilities[probabilities <= new_threshold]
            new_bad_rate = approved_loans.mean() if len(approved_loans) > 0 else 0
            
            st.metric("Simulated Approval Rate", f"{new_approval_rate:.1%}")
            st.metric("Simulated Bad Rate", f"{new_bad_rate:.3f}")
        
        with col2:
            # Threshold analysis plot
            threshold_fig = plot_threshold_analysis(explainer, X_sample)
            st.plotly_chart(threshold_fig, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Loan Default Risk System Dashboard | Built with Streamlit & XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()