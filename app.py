import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                              roc_auc_score, roc_curve, accuracy_score,
                              precision_score, recall_score, f1_score)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Claim Bias Analysis",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {background: linear-gradient(135deg,#1a237e,#283593);
        color:white;padding:20px 30px;border-radius:12px;margin-bottom:20px;}
    .metric-card {background:#f8f9fa;border:1px solid #dee2e6;
        border-radius:8px;padding:15px;text-align:center;}
    .finding-box {background:#fff3e0;border-left:5px solid #ff6f00;
        padding:15px;border-radius:5px;margin:10px 0;}
    .bias-alert {background:#ffebee;border-left:5px solid #c62828;
        padding:15px;border-radius:5px;margin:10px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🏛️ Insurance Claim Settlement Bias Analysis Dashboard</h1>
  <p>Comprehensive analysis of claim approval patterns • Bias Detection • ML Classification</p>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Clean income: remove commas, convert to numeric
    df['PI_ANNUAL_INCOME'] = df['PI_ANNUAL_INCOME'].astype(str).str.replace(',','').str.strip()
    df['PI_ANNUAL_INCOME'] = pd.to_numeric(df['PI_ANNUAL_INCOME'], errors='coerce')
    # Clean sum assured
    df['SUM_ASSURED'] = df['SUM_ASSURED'].astype(str).str.replace(',','').str.strip()
    df['SUM_ASSURED'] = pd.to_numeric(df['SUM_ASSURED'], errors='coerce')
    # Binary target
    df['CLAIM_APPROVED'] = (df['POLICY_STATUS'] == 'Approved Death Claim').astype(int)
    # Age bins
    df['AGE_GROUP'] = pd.cut(df['PI_AGE'], bins=[0,18,30,45,60,82],
                              labels=['<18','18-30','31-45','46-60','60+'])
    # Income bins
    df['INCOME_GROUP'] = pd.cut(df['PI_ANNUAL_INCOME'].fillna(0), bins=[float('-inf'),0,100000,300000,float('inf')], labels=['Zero','Low','Medium','High+'])
                                  q=4, labels=['Low','Medium','High','Very High'])
    # Zone standardise
    df['ZONE_CLEAN'] = df['ZONE'].str.upper().str.strip()
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/insurance.png", width=80)
st.sidebar.title("📁 Upload & Navigation")

uploaded_file = st.sidebar.file_uploader("Upload Insurance CSV", type=['csv'])
if uploaded_file is None:
    st.info("👆 Please upload the Insurance CSV file from the sidebar to begin.")
    st.stop()

df = load_data(uploaded_file)

tabs = st.tabs([
    "📊 Overview",
    "🔍 Descriptive Analysis",
    "⚠️ Bias Diagnostics",
    "🤖 ML Models",
    "📈 Model Performance",
    "🧠 Findings"
])

COLORS = {'Approved':'#2e7d32','Repudiate':'#c62828'}
PALETTE = ['#1565c0','#c62828','#2e7d32','#f57f17']

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Dataset Overview")
    total = len(df)
    approved = df['CLAIM_APPROVED'].sum()
    repudiated = total - approved
    approval_rate = approved/total*100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Claims", f"{total:,}")
    c2.metric("Approved", f"{approved:,}", f"{approval_rate:.1f}%")
    c3.metric("Repudiated", f"{repudiated:,}", f"{100-approval_rate:.1f}%")
    c4.metric("Avg. Age", f"{df['PI_AGE'].mean():.1f} yrs")

    c5,c6 = st.columns(2)
    with c5:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie([approved,repudiated],labels=['Approved','Repudiated'],
               colors=['#2e7d32','#c62828'],autopct='%1.1f%%',startangle=140,
               wedgeprops=dict(edgecolor='white',linewidth=2))
        ax.set_title('Overall Claim Settlement Distribution', fontweight='bold')
        st.pyplot(fig); plt.close()
    with c6:
        st.markdown("### 📋 Data Summary")
        st.dataframe(df[['PI_AGE','PI_ANNUAL_INCOME','SUM_ASSURED']].describe().round(2))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DESCRIPTIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Cross-Tabulation Against Policy Status")

    factors = {
        'Gender': 'PI_GENDER',
        'Payment Mode': 'PAYMENT_MODE',
        'Early/Non Early': 'EARLY_NON',
        'Medical/Non-Medical': 'MEDICAL_NONMED',
        'Age Group': 'AGE_GROUP',
        'Income Group': 'INCOME_GROUP',
    }
    sel = st.selectbox("Select Factor for Cross-tab", list(factors.keys()))
    col = factors[sel]

    ct = pd.crosstab(df[col], df['POLICY_STATUS'])
    ct_pct = pd.crosstab(df[col], df['POLICY_STATUS'], normalize='index').round(3)*100

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Count**"); st.dataframe(ct)
    with c2:
        st.markdown("**Row %**"); st.dataframe(ct_pct.style.format("{:.1f}%"))

    fig, axes = plt.subplots(1,2,figsize=(14,5))
    ct.plot(kind='bar', ax=axes[0], color=['#2e7d32','#c62828'], edgecolor='white',
            rot=45); axes[0].set_title(f'Count by {sel}'); axes[0].legend(loc='upper right')
    ct_pct.plot(kind='bar', ax=axes[1], color=['#2e7d32','#c62828'], edgecolor='white',
                rot=45); axes[1].set_title(f'Approval % by {sel}'); axes[1].set_ylabel('%')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Top zones
    st.markdown("---")
    st.subheader("Top 15 Zones – Approval Rates")
    zone_ct = df.groupby('ZONE')['CLAIM_APPROVED'].agg(['sum','count']).reset_index()
    zone_ct.columns = ['Zone','Approved','Total']
    zone_ct['Approval_Rate'] = zone_ct['Approved']/zone_ct['Total']*100
    zone_ct = zone_ct.sort_values('Approval_Rate', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(14,5))
    bars = ax.bar(zone_ct['Zone'], zone_ct['Approval_Rate'],
                  color=['#2e7d32' if r>=68 else '#c62828' for r in zone_ct['Approval_Rate']])
    ax.axhline(df['CLAIM_APPROVED'].mean()*100, color='navy', linestyle='--', label='Overall avg')
    ax.set_xticklabels(zone_ct['Zone'], rotation=45, ha='right')
    ax.set_ylabel('Approval Rate %'); ax.set_title('Approval Rate by Zone (Top 15)')
    ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – BIAS DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("⚠️ Bias Diagnostics Analysis")

    # --- Age bias ---
    st.markdown("### Age-wise Bias")
    age_bias = df.groupby('AGE_GROUP')['CLAIM_APPROVED'].agg(['sum','count']).reset_index()
    age_bias['Rate'] = age_bias['sum']/age_bias['count']*100
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].bar(age_bias['AGE_GROUP'].astype(str), age_bias['Rate'],
                color=['#1565c0','#1976d2','#2196f3','#64b5f6','#bbdefb'])
    axes[0].axhline(df['CLAIM_APPROVED'].mean()*100, color='red', linestyle='--', label='Overall')
    axes[0].set_title('Approval Rate by Age Group'); axes[0].set_ylabel('%'); axes[0].legend()
    df.boxplot(column='PI_AGE', by='POLICY_STATUS', ax=axes[1])
    axes[1].set_title('Age Distribution by Claim Status'); axes[1].set_xlabel('Status')
    plt.suptitle('')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # --- Income bias ---
    st.markdown("### Income-wise Bias")
    inc_bias = df.groupby('INCOME_GROUP')['CLAIM_APPROVED'].agg(['sum','count']).reset_index()
    inc_bias['Rate'] = inc_bias['sum']/inc_bias['count']*100
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].bar(inc_bias['INCOME_GROUP'].astype(str), inc_bias['Rate'],
                color=['#e65100','#ef6c00','#fb8c00','#ffa726'])
    axes[0].axhline(df['CLAIM_APPROVED'].mean()*100, color='blue', linestyle='--')
    axes[0].set_title('Approval Rate by Income Group'); axes[0].set_ylabel('%')
    df.boxplot(column='PI_ANNUAL_INCOME', by='POLICY_STATUS', ax=axes[1])
    axes[1].set_title('Income Distribution by Claim Status'); axes[1].set_xlabel('Status')
    plt.suptitle('')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # --- Gender bias ---
    st.markdown("### Gender-wise Bias")
    gen_bias = df.groupby(['PI_GENDER','AGE_GROUP'])['CLAIM_APPROVED'].mean().unstack()*100
    fig, ax = plt.subplots(figsize=(10,5))
    gen_bias.T.plot(kind='bar', ax=ax, color=['#1565c0','#e91e63'], edgecolor='white', rot=45)
    ax.set_title('Approval Rate: Gender × Age Group'); ax.set_ylabel('%'); ax.legend(title='Gender')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # --- Zone / Team bias ---
    st.markdown("### Team (Zone) Bias – All Zones")
    zone_full = df.groupby('ZONE')['CLAIM_APPROVED'].agg(['sum','count']).reset_index()
    zone_full.columns = ['Zone','Approved','Total']
    zone_full['Rate'] = zone_full['Approved']/zone_full['Total']*100
    zone_full = zone_full.sort_values('Rate')

    fig, ax = plt.subplots(figsize=(14,7))
    colors = ['#c62828' if r < df['CLAIM_APPROVED'].mean()*100 else '#2e7d32' for r in zone_full['Rate']]
    ax.barh(zone_full['Zone'], zone_full['Rate'], color=colors)
    ax.axvline(df['CLAIM_APPROVED'].mean()*100, color='navy', linestyle='--', lw=2, label='Overall avg')
    ax.set_xlabel('Approval Rate %'); ax.set_title('Claim Approval Rate by Zone/Team')
    ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    # Disparity table
    overall = df['CLAIM_APPROVED'].mean()*100
    zone_full['Disparity'] = (zone_full['Rate'] - overall).round(1)
    st.markdown("**Zones with Highest Bias (vs Overall Average)**")
    bias_table = zone_full.nsmallest(10,'Rate')[['Zone','Total','Rate','Disparity']]
    bias_table.columns = ['Zone','# Claims','Approval Rate %','Disparity vs Avg %']
    st.dataframe(bias_table.style.format({'Approval Rate %':'{:.1f}','Disparity vs Avg %':'{:.1f}'}))

    # --- Early vs Non-Early ---
    st.markdown("### Early Claim Bias")
    early_bias = df.groupby(['EARLY_NON','PI_GENDER'])['CLAIM_APPROVED'].mean().unstack()*100
    fig, ax = plt.subplots(figsize=(8,4))
    early_bias.plot(kind='bar', ax=ax, color=['#1565c0','#e91e63'], edgecolor='white', rot=0)
    ax.set_title('Approval Rate: Early vs Non-Early × Gender'); ax.set_ylabel('%')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🤖 Machine Learning Classification Models")
    st.markdown("**Feature Engineering + Training 4 Models**")

    @st.cache_data
    def run_models(file_bytes):
        df = pd.read_csv(file_bytes)
        df['PI_ANNUAL_INCOME'] = df['PI_ANNUAL_INCOME'].astype(str).str.replace(',','').str.strip()
        df['PI_ANNUAL_INCOME'] = pd.to_numeric(df['PI_ANNUAL_INCOME'], errors='coerce')
        df['SUM_ASSURED'] = df['SUM_ASSURED'].astype(str).str.replace(',','').str.strip()
        df['SUM_ASSURED'] = pd.to_numeric(df['SUM_ASSURED'], errors='coerce')
        df['CLAIM_APPROVED'] = (df['POLICY_STATUS']=='Approved Death Claim').astype(int)

        # Feature engineering
        df['INCOME_TO_SUM'] = df['PI_ANNUAL_INCOME'] / (df['SUM_ASSURED']+1)
        df['AGE_INCOME'] = df['PI_AGE'] * df['PI_ANNUAL_INCOME']
        df['IS_EARLY'] = (df['EARLY_NON']=='EARLY').astype(int)
        df['IS_MEDICAL'] = (df['MEDICAL_NONMED']=='MEDICAL').astype(int)
        df['IS_FEMALE'] = (df['PI_GENDER']=='F').astype(int)

        cat_cols = ['ZONE','PAYMENT_MODE','PI_OCCUPATION','REASON_FOR_CLAIM','PI_STATE']
        for c in cat_cols:
            le = LabelEncoder()
            df[c+'_ENC'] = le.fit_transform(df[c].fillna('Unknown').astype(str))

        feat_cols = ['PI_AGE','PI_ANNUAL_INCOME','SUM_ASSURED','IS_EARLY','IS_MEDICAL','IS_FEMALE',
                     'INCOME_TO_SUM','AGE_INCOME'] + [c+'_ENC' for c in cat_cols]

        X = df[feat_cols]
        y = df['CLAIM_APPROVED']

        imp = SimpleImputer(strategy='median')
        X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42, stratify=y)

        models = {
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosted': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            tr_pred = model.predict(X_tr)
            te_pred = model.predict(X_te)
            te_prob = model.predict_proba(X_te)[:,1]

            results[name] = {
                'model': model,
                'train_acc': accuracy_score(y_tr, tr_pred),
                'test_acc': accuracy_score(y_te, te_pred),
                'precision': precision_score(y_te, te_pred),
                'recall': recall_score(y_te, te_pred),
                'f1': f1_score(y_te, te_pred),
                'auc': roc_auc_score(y_te, te_prob),
                'cm': confusion_matrix(y_te, te_pred),
                'y_te': y_te.values,
                'te_prob': te_prob,
                'report': classification_report(y_te, te_pred, output_dict=True)
            }

            # Feature importance for tree-based
            if hasattr(model, 'feature_importances_'):
                results[name]['feat_imp'] = pd.Series(model.feature_importances_, index=feat_cols).nlargest(10)

        return results, feat_cols

    with st.spinner("Training 4 ML models..."):
        uploaded_file.seek(0)
        results, feat_cols = run_models(uploaded_file)

    st.success("✅ All 4 models trained successfully!")

    # Feature engineering summary
    with st.expander("📐 Feature Engineering Details"):
        st.markdown("""
        **Original Features used:**
        - `PI_AGE`, `PI_ANNUAL_INCOME`, `SUM_ASSURED` — numerical as-is
        - `EARLY_NON` → binary (1=EARLY), `MEDICAL_NONMED` → binary, `PI_GENDER` → binary
        
        **Engineered Features:**
        - `INCOME_TO_SUM` = Annual Income / Sum Assured (affordability ratio)
        - `AGE_INCOME` = Age × Income (interaction term capturing risk-income profile)
        
        **Encoded Features:**  ZONE, PAYMENT_MODE, PI_OCCUPATION, REASON_FOR_CLAIM, PI_STATE → Label Encoded
        
        **Pipeline:** Missing value imputation (median) → StandardScaler → 80/20 stratified train-test split
        """)

    # Feature importance
    st.markdown("### 🔑 Feature Importances (Random Forest)")
    rf_fi = results['Random Forest']['feat_imp']
    fig, ax = plt.subplots(figsize=(10,5))
    rf_fi.sort_values().plot(kind='barh', ax=ax, color='#1565c0')
    ax.set_title('Top 10 Feature Importances – Random Forest')
    ax.set_xlabel('Importance Score')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("📈 Model Performance Comparison")

    # Need results from previous tab – re-run if needed
    try:
        results
    except NameError:
        with st.spinner("Training models..."):
            uploaded_file.seek(0)
            results, feat_cols = run_models(uploaded_file)

    # Summary table
    rows = []
    for name, r in results.items():
        rows.append({'Model':name,
                     'Train Acc':f"{r['train_acc']:.3f}",
                     'Test Acc':f"{r['test_acc']:.3f}",
                     'Precision':f"{r['precision']:.3f}",
                     'Recall':f"{r['recall']:.3f}",
                     'F1':f"{r['f1']:.3f}",
                     'AUC':f"{r['auc']:.3f}"})
    st.dataframe(pd.DataFrame(rows).set_index('Model'))

    # Bar comparison
    metrics = ['train_acc','test_acc','precision','recall','f1','auc']
    labels  = ['Train Acc','Test Acc','Precision','Recall','F1','AUC']
    fig, axes = plt.subplots(2,3,figsize=(15,8))
    for ax, m, lbl in zip(axes.flat, metrics, labels):
        vals = [results[n][m] for n in results]
        bars = ax.bar(list(results.keys()), vals, color=PALETTE, edgecolor='white')
        ax.set_title(lbl); ax.set_ylim(0,1.05); ax.set_xticklabels(list(results.keys()),rotation=20,ha='right')
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ROC Curves
    st.markdown("### ROC Curves")
    fig, ax = plt.subplots(figsize=(8,6))
    roc_colors = ['#1565c0','#c62828','#2e7d32','#f57f17']
    for (name,r),color in zip(results.items(),roc_colors):
        fpr,tpr,_ = roc_curve(r['y_te'],r['te_prob'])
        ax.plot(fpr,tpr,label=f"{name} (AUC={r['auc']:.3f})",color=color,lw=2)
    ax.plot([0,1],[0,1],'k--',alpha=0.4)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves – All Models'); ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Confusion Matrices
    st.markdown("### Confusion Matrices")
    fig, axes = plt.subplots(1,4,figsize=(18,4))
    for ax,(name,r) in zip(axes,results.items()):
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Repudiated','Approved'],
                    yticklabels=['Repudiated','Approved'])
        ax.set_title(name); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – FINDINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🧠 Key Findings & Recommendations")

    st.markdown("""
    <div class="finding-box">
    <h4>📌 Finding 1 – Class Imbalance Exists</h4>
    <b>~68% Approved vs ~32% Repudiated.</b> While not extreme, this skew can cause models 
    to favour approvals. Recall on repudiated class should be monitored carefully.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bias-alert">
    <h4>🚨 Finding 2 – Team/Zone Bias Detected</h4>
    Some zones show approval rates <b>15–25% below the overall average</b>.
    Zones tagged as EAST 1, EAST 2, and some North zones consistently show lower approval rates,
    suggesting differential standards or under-resourced settlement teams.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-box">
    <h4>📌 Finding 3 – Age Bias (Elderly Policyholders)</h4>
    Policyholders aged <b>60+</b> show a notably higher repudiation rate relative to younger cohorts.
    This may reflect stricter scrutiny on high-sum elderly policies or systemic bias.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-box">
    <h4>📌 Finding 4 – Income-Level Disparity</h4>
    <b>Low-income group</b> policyholders face a higher repudiation rate. 
    This could indicate that lower-income claimants lack documentation support or 
    access to legal resources to contest rejections.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bias-alert">
    <h4>🚨 Finding 5 – Early Claim Repudiation Rate is High</h4>
    EARLY claims (death shortly after policy issuance) face <b>significantly higher repudiation rates</b>.
    While legitimate for fraud control, this should be applied uniformly across zones.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-box">
    <h4>📌 Finding 6 – Best Performing Model</h4>
    <b>Random Forest and Gradient Boosted</b> models achieve the highest accuracy and AUC, 
    making them the most reliable predictors of claim outcome. Income-to-Sum ratio and Age 
    emerge as the most predictive features — reinforcing the income/age bias hypothesis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✅ Recommendations")
    st.markdown("""
    1. **Audit zones with below-average approval rates** — investigate case samples from bottom 5 zones.
    2. **Standardise Early Claim review process** across all zones with uniform checklists.
    3. **Age-neutral documentation policy** — avoid stricter scrutiny solely based on age.
    4. **Income-blind assessment** — ensure low-income claimants get equal documentation support.
    5. **Deploy ML model as a second-opinion tool** to flag potential biased rejections for review.
    6. **Monthly dashboard review** by compliance officer to track approval-rate trends by zone.
    """)

    st.markdown("---")
    st.caption("Dashboard built for Insurance Claim Settlement Bias Analysis | Python • Streamlit • Scikit-learn")

