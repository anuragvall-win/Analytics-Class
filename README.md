# Insurance Claim Settlement Bias Analysis Dashboard

## Overview
A Streamlit dashboard to analyse bias in insurance claim settlement decisions.
Covers descriptive analysis, bias diagnostics, and 4 ML classifiers.

## Features
- 📊 Overview & dataset statistics
- 🔍 Cross-tabulation analysis by gender, age, income, zone, payment mode
- ⚠️ Bias diagnostics: age-wise, income-wise, gender-wise, team/zone-wise
- 🤖 ML Models: KNN, Decision Tree, Random Forest, Gradient Boosted
- 📈 Model performance: accuracy, precision, recall, F1, ROC, confusion matrix
- 🧠 Findings & recommendations

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Free)

1. Fork/push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Click **New app**
4. Select your repo, branch (`main`) and file (`app.py`)
5. Click **Deploy**
6. Upload your CSV from the sidebar when the app loads

## File Structure
```
insurance_dashboard/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Usage
1. Run the app (locally or on Streamlit Cloud)
2. Upload your `Insurance.csv` from the sidebar
3. Navigate through the 6 tabs
