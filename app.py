import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from fpdf import FPDF
from groq import Groq

st.set_page_config(page_title="Cleanora AI", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css?family=Ubuntu:700|lato:400');

    :root {
      --text: #e0e0ff;
      --background: #0b0f19;
      --primary: #844dfc;
      --secondary: #1a233a;
      --accent: #ff8c42;
    }

    html, body, p, label {
      font-family: 'lato', sans-serif !important;
      color: #e0e0ff !important;
    }
    
    /* Protect Material Icons from being overridden by custom fonts */
    .material-symbols-rounded, span[data-testid="stIconMaterial"], .material-icons {
      font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }
    
    .stApp {
      background-color: #0b0f19;
      background-image: 
        radial-gradient(circle at 10% 20%, rgba(132, 77, 252, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(255, 140, 66, 0.1) 0%, transparent 40%);
    }

    h1, h2, h3, h4, h5 {
      font-family: 'Ubuntu', sans-serif !important;
      font-weight: 700 !important;
      color: #ffffff !important;
    }
    
    h1 { font-size: 4.210rem !important; }
    h2 { font-size: 3.158rem !important; }
    h3 { font-size: 2.369rem !important; }
    h4 { font-size: 1.777rem !important; }
    h5 { font-size: 1.333rem !important; }

    [data-testid="stSidebar"] {
      background: rgba(11, 15, 25, 0.6) !important;
      backdrop-filter: blur(15px);
      border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button {
      background: linear-gradient(135deg, #844dfc, #ff8c42);
      color: white !important;
      font-family: 'Ubuntu', sans-serif !important;
      border: none;
      border-radius: 25px;
      padding: 0.5rem 1.5rem;
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(132, 77, 252, 0.3);
    }
    .stButton > button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(255, 140, 66, 0.4);
    }

    [data-testid="stMetric"] {
      background: rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(10px);
      border-radius: 15px;
      padding: 1.2rem;
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    [data-testid="stMetricValue"] {
      color: #ff8c42 !important;
      font-family: 'Ubuntu', sans-serif !important;
      font-weight: 700 !important;
    }
    
    [data-testid="stDataFrame"] {
      border-radius: 15px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    [data-testid="stFileUploader"] {
      background: rgba(255, 255, 255, 0.05);
      border-radius: 15px;
      padding: 1rem;
      border: 2px dashed #844dfc;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(132, 77, 252, 0.2);
        border-bottom-color: #ff8c42;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Cleanora AI")
st.markdown("Automated, AI-powered dataset cleaning and preprocessing for analytics.")

# State initialization
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'cleaned_df' not in st.session_state: st.session_state.cleaned_df = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'cleaning_log' not in st.session_state: st.session_state.cleaning_log = []
if 'current_file_name' not in st.session_state: st.session_state.current_file_name = None
if 'history_dfs' not in st.session_state: st.session_state.history_dfs = []
if 'history_logs' not in st.session_state: st.session_state.history_logs = []

def save_state():
    if st.session_state.cleaned_df is not None:
        st.session_state.history_dfs.append(st.session_state.cleaned_df.copy())
        st.session_state.history_logs.append(list(st.session_state.cleaning_log))

def undo_action():
    if st.session_state.history_dfs:
        st.session_state.cleaned_df = st.session_state.history_dfs.pop()
        st.session_state.cleaning_log = st.session_state.history_logs.pop()

def calc_quality_score(df):
    if df is None or df.empty: return 0
    total_cells = df.size
    if total_cells == 0: return 0
    missing = df.isna().sum().sum()
    dupes = df.duplicated().sum() * df.shape[1]
    issues = missing + dupes
    score = max(0, min(100, 100 - (issues / total_cells * 100)))
    return round(score, 1)

def export_pdf_report(logs, initial_score, final_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15, style='B')
    pdf.cell(200, 10, txt="Cleanora AI - Cleaning Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Initial Data Quality Score: {initial_score}%", ln=True)
    pdf.cell(200, 10, txt=f"Final Data Quality Score: {final_score}%", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, txt="Actions Applied:", ln=True)
    pdf.set_font("Arial", size=10)
    for log in logs:
        # Encode string to latin-1 and replace unknown chars to avoid FPDF errors
        log_text = str(log).encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(200, 10, txt=f"- {log_text}", ln=True)
    
    # Save to a byte string
    return pdf.output(dest='S').encode('latin-1')

# Sidebar
with st.sidebar:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose CSV/Excel/JSON", type=["csv", "xlsx", "json"])
    
    st.markdown("---")
    st.header("AI Settings")
    groq_api_key = st.text_input("Groq API Key (Optional)", type="password")
    
    st.markdown("---")
    if st.button("Generate Sample Data"):
        np.random.seed(42)
        sample_size = 500
        dates = pd.date_range(start="2023-01-01", periods=sample_size)
        categories = np.random.choice(["Electronics", "Clothing", "Home", "Sports", None], size=sample_size, p=[0.3, 0.3, 0.2, 0.15, 0.05])
        prices = np.random.normal(loc=100, scale=40, size=sample_size)
        prices[np.random.choice(sample_size, size=20, replace=False)] = np.nan 
        prices[np.random.choice(sample_size, size=5, replace=False)] = 5000 
        
        df_sample = pd.DataFrame({
            "Date": dates,
            "Category": categories,
            "Price": prices,
            "Quantity": np.random.randint(1, 10, size=sample_size)
        })
        df_sample = pd.concat([df_sample, df_sample.sample(15)])
        df_sample.reset_index(drop=True, inplace=True)
        st.session_state.raw_df = df_sample
        st.session_state.cleaned_df = df_sample.copy()
        st.session_state.cleaning_log = ["Sample data generated."]
        st.session_state.current_file_name = "sample_data"
        st.session_state.history_dfs = []
        st.session_state.history_logs = []
        st.success("Sample Data Loaded!")

if uploaded_file is not None and st.session_state.current_file_name != uploaded_file.name:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.raw_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            st.session_state.raw_df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            st.session_state.raw_df = pd.read_json(uploaded_file)
        
        st.session_state.cleaned_df = st.session_state.raw_df.copy()
        st.session_state.cleaning_log = ["Dataset loaded."]
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.history_dfs = []
        st.session_state.history_logs = []
        st.rerun()
    except Exception as e:
        st.error(f"Error loading file: {e}")

if st.session_state.cleaned_df is not None:
    raw_df = st.session_state.raw_df
    cleaned_df = st.session_state.cleaned_df
    
    tab_dash, tab_clean, tab_analy, tab_ai = st.tabs(["Dashboard", "Interactive Cleaning", "Analytics", "AI Assistant"])
    
    # ---------------- TAB 1: DASHBOARD ----------------
    with tab_dash:
        st.header("Dataset Quality Overview")
        
        score_orig = calc_quality_score(raw_df)
        score_curr = calc_quality_score(cleaned_df)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Original Quality Score", f"{score_orig}%")
        col_s2.metric("Current Quality Score", f"{score_curr}%", f"{round(score_curr - score_orig, 1)}%")
        col_s3.metric("Fixes Applied", len(st.session_state.cleaning_log) - 1)
        
        st.subheader("Data Preview")
        toggle_view = st.radio("View:", ["Cleaned Data", "Original Data"], horizontal=True)
        if toggle_view == "Cleaned Data":
            st.dataframe(cleaned_df.head(50), use_container_width=True)
        else:
            st.dataframe(raw_df.head(50), use_container_width=True)
            
        st.markdown("---")
        # Export Actions
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_data = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Cleaned CSV", data=csv_data, file_name="cleaned_dataset.csv", mime="text/csv")
        with col_exp2:
            try:
                pdf_bytes = export_pdf_report(st.session_state.cleaning_log, score_orig, score_curr)
                st.download_button("Download PDF Report", data=pdf_bytes, file_name="cleaning_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF generation error: {e}. Note: fpdf might have font limitations with certain characters.")

    # ---------------- TAB 2: INTERACTIVE CLEANING ----------------
    with tab_clean:
        st.header("Interactive Cleaning Tools")
        
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Auto-Clean Dataset"):
                save_state()
                # Drop Duplicates
                dups = cleaned_df.duplicated().sum()
                cleaned_df.drop_duplicates(inplace=True)
                if dups > 0: st.session_state.cleaning_log.append(f"Auto-Clean: Removed {dups} duplicate rows.")
                
                # Impute Missing
                for col in cleaned_df.columns:
                    if cleaned_df[col].isna().sum() > 0:
                        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                        else:
                            mode_val = cleaned_df[col].mode()
                            if not mode_val.empty:
                                cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
                            else:
                                cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                st.session_state.cleaning_log.append("Auto-Clean: Imputed missing values.")
                
                st.session_state.cleaned_df = cleaned_df
                st.rerun()

        with action_col2:
            if st.button(f"Undo Last Action ({len(st.session_state.history_dfs)})", disabled=len(st.session_state.history_dfs)==0):
                undo_action()
                st.rerun()
                
        st.markdown("---")
        clean_col1, clean_col2 = st.columns(2)
        
        # 1. Missing Values
        with clean_col1:
            with st.expander("Missing Value Detection", expanded=True):
                missing_counts = cleaned_df.isna().sum()
                missing_cols = missing_counts[missing_counts > 0].index.tolist()
                
                if not missing_cols:
                    st.success("No missing values detected!")
                else:
                    sel_col = st.selectbox("Select Column to Impute", missing_cols)
                    st.write(f"Missing Values: {missing_counts[sel_col]}")
                    method = st.selectbox("Imputation Method", ["Mean", "Median", "Mode", "Forward Fill"])
                    
                    if st.button("Apply Fix"):
                        save_state()
                        if method == "Mean" and pd.api.types.is_numeric_dtype(cleaned_df[sel_col]):
                            cleaned_df[sel_col] = cleaned_df[sel_col].fillna(cleaned_df[sel_col].mean())
                            st.session_state.cleaning_log.append(f"Filled missing values in {sel_col} with Mean.")
                        elif method == "Median" and pd.api.types.is_numeric_dtype(cleaned_df[sel_col]):
                            cleaned_df[sel_col] = cleaned_df[sel_col].fillna(cleaned_df[sel_col].median())
                            st.session_state.cleaning_log.append(f"Filled missing values in {sel_col} with Median.")
                        elif method == "Mode":
                            mode_val = cleaned_df[sel_col].mode()
                            if not mode_val.empty:
                                cleaned_df[sel_col] = cleaned_df[sel_col].fillna(mode_val[0])
                                st.session_state.cleaning_log.append(f"Filled missing values in {sel_col} with Mode.")
                        elif method == "Forward Fill":
                            cleaned_df[sel_col] = cleaned_df[sel_col].ffill()
                            st.session_state.cleaning_log.append(f"Filled missing values in {sel_col} using Forward Fill.")
                        else:
                            st.error("Method not applicable to this column type.")
                        st.session_state.cleaned_df = cleaned_df
                        st.rerun()

        # 2. Duplicate Removal
        with clean_col2:
            with st.expander("Duplicate Removal", expanded=True):
                dupes = cleaned_df.duplicated().sum()
                st.write(f"Duplicate Rows: **{dupes}**")
                if dupes > 0:
                    if st.button("Remove Duplicates"):
                        save_state()
                        st.session_state.cleaned_df = cleaned_df.drop_duplicates()
                        st.session_state.cleaning_log.append(f"Removed {dupes} duplicate rows.")
                        st.rerun()
                else:
                    st.success("No duplicates found!")

        # 3. Datatype Correction
        with clean_col1:
            with st.expander("Datatype Correction"):
                type_col = st.selectbox("Select Column", cleaned_df.columns, key='type_col')
                current_type = cleaned_df[type_col].dtype
                st.write(f"Current Type: `{current_type}`")
                target_type = st.selectbox("Convert To", ["Numeric", "String", "Datetime"])
                if st.button("Convert Datatype"):
                    save_state()
                    try:
                        if target_type == "Numeric":
                            st.session_state.cleaned_df[type_col] = pd.to_numeric(cleaned_df[type_col], errors='coerce')
                        elif target_type == "String":
                            st.session_state.cleaned_df[type_col] = cleaned_df[type_col].astype(str)
                        elif target_type == "Datetime":
                            st.session_state.cleaned_df[type_col] = pd.to_datetime(cleaned_df[type_col], errors='coerce')
                        st.session_state.cleaning_log.append(f"Converted {type_col} to {target_type}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Conversion failed: {e}")

        # 4. Normalization
        with clean_col2:
            with st.expander("Data Normalization"):
                num_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
                if not num_cols:
                    st.info("No numeric columns available.")
                else:
                    norm_col = st.selectbox("Select Numeric Column", num_cols, key='norm_col')
                    norm_method = st.selectbox("Method", ["Min-Max Scaling", "Standard Scaling", "Z-score Normalization"])
                    if st.button("Normalize"):
                        save_state()
                        if norm_method == "Min-Max Scaling":
                            scaler = MinMaxScaler()
                            st.session_state.cleaned_df[[norm_col]] = scaler.fit_transform(cleaned_df[[norm_col]])
                        else:
                            scaler = StandardScaler()
                            st.session_state.cleaned_df[[norm_col]] = scaler.fit_transform(cleaned_df[[norm_col]])
                        st.session_state.cleaning_log.append(f"Normalized {norm_col} using {norm_method}.")
                        st.rerun()

    # ---------------- TAB 3: ANALYTICS ----------------
    with tab_analy:
        st.header("Dataset Analytics")
        
        # Heatmap
        st.subheader("Missing Value Heatmap")
        fig_heat, ax_heat = plt.subplots(figsize=(10, 4))
        # Set dark theme for matplotlib
        plt.style.use('dark_background')
        fig_heat.patch.set_facecolor('#0b0f19')
        ax_heat.set_facecolor('#0b0f19')
        sns.heatmap(cleaned_df.isna(), yticklabels=False, cbar=False, cmap='viridis', ax=ax_heat)
        st.pyplot(fig_heat)

        # Columns Summary
        st.subheader("Column Datatype Summary")
        type_summary = pd.DataFrame(cleaned_df.dtypes, columns=['Datatype']).astype(str)
        st.dataframe(type_summary.T, use_container_width=True)

        # Statistics
        st.subheader("Dataset Statistics")
        st.dataframe(cleaned_df.describe(), use_container_width=True)

    # ---------------- TAB 4: AI ASSISTANT ----------------
    with tab_ai:
        st.header("AI Cleaning Assistant")
        st.markdown("Chat with Groq-powered AI to analyze and clean your data.")
        
        if not groq_api_key:
            st.warning("Please enter your Groq API Key in the sidebar to enable the AI Assistant.")
        else:
            try:
                client = Groq(api_key=groq_api_key)
                
                # Display chat
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
                
                # Chat input
                prompt = st.chat_input("E.g., 'What should I do with the missing values in Price?'")
                if prompt:
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    with st.chat_message("assistant"):
                        # Context building
                        col_info = cleaned_df.dtypes.to_dict()
                        missing_info = cleaned_df.isna().sum().to_dict()
                        sys_prompt = f"""You are a Data Cleaning AI Assistant. The user has a dataset with {cleaned_df.shape[0]} rows and {cleaned_df.shape[1]} columns.
                        Columns: {list(cleaned_df.columns)}. 
                        Missing values: {missing_info}. 
                        Help them analyze or suggest cleaning steps based on their query. Be concise."""
                        
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=300
                        )
                        msg_text = response.choices[0].message.content
                        st.write(msg_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": msg_text})
            
            except Exception as e:
                st.error(f"Error initializing AI Assistant: {e}")

else:
    # Landing view
    st.markdown("""
        <style>
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 60vh;
            text-align: center;
            background: radial-gradient(circle, rgba(132, 77, 252, 0.15) 0%, transparent 70%);
            border-radius: 20px;
            margin-top: 2rem;
        }

        .glowing-text {
            font-family: 'Ubuntu', sans-serif !important;
            font-size: 5rem;
            font-weight: 700;
            color: rgba(240, 230, 255, 0.9);
            text-transform: uppercase;
            letter-spacing: 2px;
            animation: glow 3s ease-in-out infinite alternate;
            margin-bottom: 1rem;
        }

        .subtitle {
            font-family: 'Lato', sans-serif !important;
            font-size: 1.5rem;
            color: #e0e0ff;
            opacity: 0;
            animation: fadeIn 2s ease-in forwards;
            animation-delay: 0.5s;
        }

        @keyframes glow {
            from {
                text-shadow: 0 0 30px rgba(230, 204, 255, 0.6), 0 0 60px rgba(220, 179, 255, 0.5), 0 0 90px rgba(220, 179, 255, 0.4), 0 0 120px rgba(205, 164, 255, 0.3);
            }
            to {
                text-shadow: 0 0 40px rgba(230, 204, 255, 0.8), 0 0 80px rgba(220, 179, 255, 0.6), 0 0 120px rgba(220, 179, 255, 0.5), 0 0 160px rgba(205, 164, 255, 0.4), 0 0 200px rgba(205, 164, 255, 0.2);
            }
        }

        @keyframes fadeIn {
            to {
                opacity: 1;
            }
        }
        </style>
        
        <div class="welcome-container">
            <div class="glowing-text">Welcome</div>
            <div class="subtitle">Please upload a dataset from the sidebar to begin, or click 'Generate Sample Data' to test the application.</div>
        </div>
    """, unsafe_allow_html=True)
