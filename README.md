# ✨ Cleanora AI

Welcome to **Cleanora AI** — an automated, AI-powered dataset cleaning and preprocessing tool designed for analytics and machine learning. Say goodbye to messy datasets and manual data wrangling!

## 🚀 Features

### 🛠️ Interactive Cleaning Suite
- **Auto-Clean Feature**: Click a single button to drop duplicates and intelligently impute missing values across the entire dataset!
- **Missing Value Detection & Imputation**: Easily spot null values and fix them interactively using Mean, Median, Mode, or Forward Fill methods.
- **Duplicate Removal**: Automatically detect and drop duplicate rows from your datasets.
- **Datatype Correction**: One-click conversions to Numeric, String, or Datetime types.
- **Data Normalization**: Scale your numeric data instantly using Min-Max Scaling, Standard Scaling, or Z-score normalization.
- **Undo History**: Made a mistake? Seamlessly undo your last actions and step back through your cleaning history.

### 🤖 AI Assistant (Powered by Groq)
Chat directly with our integrated AI Assistant! Just plug in your Groq API key and ask the LLM (Llama-3.1-8b-instant) to analyze your dataset, suggest cleaning methods, or help you understand missing values.

### 📈 Analytics Dashboard
- View an interactive **Missing Value Heatmap** powered by Seaborn.
- Compare data distributions before and after cleaning using interactive Plotly Box Plots.
- Check real-time **Data Quality Scores** to measure your cleaning progress.

### 💾 Export & Reporting
- Download your freshly cleaned data in CSV format.
- Generate and download a comprehensive **PDF Cleaning Report** detailing every action taken and the overall improvement in data quality.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy, Scikit-Learn
- **Visualizations**: Plotly, Matplotlib, Seaborn
- **AI Integration**: Groq
- **PDF Generation**: FPDF

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SanviScript/Cleanora-AI.git
   cd Cleanora-AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Upload your data!**
   Navigate to `http://localhost:8501`, upload your CSV, Excel, or JSON file, and let Cleanora AI handle the rest!
