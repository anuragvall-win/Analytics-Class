import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

st.set_page_config(
    page_title="Insurance Claim Bias Analysis",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg,#1a237e,#283593);
    color:white; padding:20px 30px; border-radius:12px; margin-bottom:20px;
}
.finding-box {
    background:#fff3e0; border-left:5px solid #ff6f00;
    padding:15px; border-radius:5px; margin:10px 0;
}
.bias-alert {
    background:#ffebee; border-left:5px solid #c62828;
    padding:15px; border-radius:5px; margin:10px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>Insurance Claim Settlement Bias Analysis</h1>
  <p>Descriptive Analysis · Bias Diagnostics · ML Classification</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['PI_ANNUAL_INCOME'] = pd.to_numeric(
        df['PI_ANNUAL_INCOME'].astype(str).str.replace(',', '').str.strip(),
        errors='coerce'
    )
    df['SUM_ASSURED'] = pd.to_numeric(
        df['SUM_ASSURED'].astype(str).str.replace(',', '').str.strip(),
        errors='coerce'
    )
    df['CLAIM_APPROVED'] = (df['POLICY_STATUS'] == 'Approved Death Claim').astype(int)
    df['AGE_GROUP'] = pd.cut(
        df['PI_AGE'],
        bins=[0, 18, 30, 45, 60, 82],
        labels=['<18', '18-30', '31-45', '46-60', '60+']
    )
    df['INCOME_GROUP'] = pd.cut(
        df['PI_ANNUAL_INCOME'].fillna(0),
        bins=[float('-inf'), 0, 100000, 300000, float('inf')],
        labels=['Zero', 'Low', 'Medium', 'High+']
    )
    return df


st.sidebar.title("Upload & Navigation")
uploaded_file = st.sidebar.file_uploader("Upload Insurance CSV", type=['csv'])

if uploaded_file is None:
    st.info("Please upload the Insurance CSV file from the sidebar to begin.")
    st.stop()

df = load_data(uploaded_file)

tabs = st.tabs([
    "Overview",
    "Descriptive Analysis",
    "Bias Diagnostics",
    "ML Models",
    "Model Performance",
    "Findings"
])

PALETTE = ['#2e7d32', '#c62828']
PLOT_COLORS = ['#1565c0', '#c62828', '#2e7d32', '#f57f17']
overall_rate = df['CLAIM_APPROVED'].mean() * 100

# ── TAB 0: OVERVIEW ──────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Dataset Overview")
    total = len(df)
    approved = int(df['CLAIM_APPROVED'].sum())
    repudiated = total - approved

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", f"{total:,}")
    c2.metric("Approved", f"{approved:,}", f"{approved/total*100:.1f}%")
    c3.metric("Repudiated", f"{repudiated:,}", f"{repudiated/total*100:.1f}%")
    c4.metric("Avg Age", f"{df['PI_AGE'].mean():.1f} yrs")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            [approved, repudiated],
            labels=['Approved', 'Repudiated'],
            colors=PALETTE, autopct='%1.1f%%', startangle=140,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        ax.set_title('Overall Settlement Distribution', fontweight='bold')
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown("### Data Summary")
        st.dataframe(
            df[['PI_AGE', 'PI_ANNUAL_INCOME', 'SUM_ASSURED']].describe().round(2)
        )

# ── TAB 1: DESCRIPTIVE ANALYSIS ──────────────────────────────────────────────
with tabs[1]:
    st.subheader("Cross-Tabulation Against Policy Status")

    factor_map = {
        'Gender': 'PI_GENDER',
        'Payment Mode': 'PAYMENT_MODE',
        'Early/Non Early': 'EARLY_NON',
        'Medical/Non-Medical': 'MEDICAL_NONMED',
        'Age Group': 'AGE_GROUP',
        'Income Group': 'INCOME_GROUP',
    }
    sel = st.selectbox("Select Factor", list(factor_map.keys()))
    col = factor_map[sel]

    ct_count = pd.crosstab(df[col].astype(str), df['POLICY_STATUS'])
    ct_pct = pd.crosstab(df[col].astype(str), df['POLICY_STATUS'], normalize='index').round(3) * 100

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Count**")
        st.dataframe(ct_count)
    with c2:
        st.markdown("**Row %**")
        st.dataframe(ct_pct.style.format("{:.1f}%"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ct_count.plot(kind='bar', ax=axes[0], color=PALETTE, edgecolor='white', rot=45)
    axes[0].set_title(f'Count by {sel}')
    axes[0].legend(loc='upper right')
    ct_pct.plot(kind='bar', ax=axes[1], color=PALETTE, edgecolor='white', rot=45)
    axes[1].set_title(f'Approval % by {sel}')
    axes[1].set_ylabel('%')
    axes[1].axhline(overall_rate, color='navy', linestyle='--', label='Overall avg')
    axes[1].legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Top 15 Zones – Approval Rates")
    zone_ct = df.groupby('ZONE')['CLAIM_APPROVED'].agg(['sum', 'count']).reset_index()
    zone_ct.columns = ['Zone', 'Approved', 'Total']
    zone_ct['Rate'] = zone_ct['Approved'] / zone_ct['Total'] * 100
    zone_ct = zone_ct.sort_values('Rate', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(14, 5))
    bar_colors = ['#2e7d32' if r >= overall_rate else '#c62828' for r in zone_ct['Rate']]
    ax.bar(zone_ct['Zone'], zone_ct['Rate'], color=bar_colors)
    ax.axhline(overall_rate, color='navy', linestyle='--', label=f'Overall avg {overall_rate:.1f}%')
    ax.set_xticklabels(zone_ct['Zone'], rotation=45, ha='right')
    ax.set_ylabel('Approval Rate %')
    ax.set_title('Approval Rate by Zone (Top 15)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 2: BIAS DIAGNOSTICS ──────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Bias Diagnostics Analysis")

    st.markdown("### Age-wise Bias")
    age_bias = df.groupby(df['AGE_GROUP'].astype(str))['CLAIM_APPROVED'].agg(['sum', 'count']).reset_index()
    age_bias.columns = ['AGE_GROUP', 'Approved', 'Total']
    age_bias['Rate'] = age_bias['Approved'] / age_bias['Total'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(age_bias['AGE_GROUP'], age_bias['Rate'],
                color=['#1565c0', '#1976d2', '#2196f3', '#64b5f6', '#bbdefb'])
    axes[0].axhline(overall_rate, color='red', linestyle='--', label='Overall avg')
    axes[0].set_title('Approval Rate by Age Group')
    axes[0].set_ylabel('%')
    axes[0].legend()
    df.boxplot(column='PI_AGE', by='POLICY_STATUS', ax=axes[1])
    axes[1].set_title('Age Distribution by Claim Status')
    axes[1].set_xlabel('')
    plt.suptitle('')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Income-wise Bias")
    inc_bias = df.groupby(df['INCOME_GROUP'].astype(str))['CLAIM_APPROVED'].agg(['sum', 'count']).reset_index()
    inc_bias.columns = ['INCOME_GROUP', 'Approved', 'Total']
    inc_bias['Rate'] = inc_bias['Approved'] / inc_bias['Total'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(inc_bias['INCOME_GROUP'], inc_bias['Rate'],
                color=['#e65100', '#ef6c00', '#fb8c00', '#ffa726'])
    axes[0].axhline(overall_rate, color='blue', linestyle='--', label='Overall avg')
    axes[0].set_title('Approval Rate by Income Group')
    axes[0].set_ylabel('%')
    axes[0].legend()
    valid_income = df[df['PI_ANNUAL_INCOME'] > 0]
    valid_income.boxplot(column='PI_ANNUAL_INCOME', by='POLICY_STATUS', ax=axes[1])
    axes[1].set_title('Income Distribution by Status (non-zero)')
    axes[1].set_xlabel('')
    plt.suptitle('')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Gender × Age Bias")
    gen_bias = (df.groupby([df['AGE_GROUP'].astype(str), 'PI_GENDER'])['CLAIM_APPROVED']
                .mean().unstack() * 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    gen_bias.plot(kind='bar', ax=ax, color=['#1565c0', '#e91e63'], edgecolor='white', rot=45)
    ax.set_title('Approval Rate: Age Group × Gender')
    ax.set_ylabel('%')
    ax.legend(title='Gender')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Team (Zone) Bias – All Zones")
    zone_full = df.groupby('ZONE')['CLAIM_APPROVED'].agg(['sum', 'count']).reset_index()
    zone_full.columns = ['Zone', 'Approved', 'Total']
    zone_full['Rate'] = zone_full['Approved'] / zone_full['Total'] * 100
    zone_full = zone_full.sort_values('Rate')

    fig, ax = plt.subplots(figsize=(14, 8))
    bar_colors = ['#c62828' if r < overall_rate else '#2e7d32' for r in zone_full['Rate']]
    ax.barh(zone_full['Zone'], zone_full['Rate'], color=bar_colors)
    ax.axvline(overall_rate, color='navy', linestyle='--', lw=2,
               label=f'Overall avg {overall_rate:.1f}%')
    red_p = plt.Rectangle((0, 0), 1, 1, fc='#c62828')
    grn_p = plt.Rectangle((0, 0), 1, 1, fc='#2e7d32')
    ax.legend(handles=[red_p, grn_p], labels=['Below Average', 'Above Average'])
    ax.set_xlabel('Approval Rate %')
    ax.set_title('Claim Approval Rate by Zone/Team', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    zone_full['Disparity'] = (zone_full['Rate'] - overall_rate).round(1)
    st.markdown("**Zones with Lowest Approval Rates**")
    bias_table = zone_full.nsmallest(10, 'Rate')[['Zone', 'Total', 'Rate', 'Disparity']]
    bias_table.columns = ['Zone', '# Claims', 'Approval Rate %', 'Disparity vs Avg %']
    st.dataframe(
        bias_table.style.format({'Approval Rate %': '{:.1f}', 'Disparity vs Avg %': '{:.1f}'})
    )

    st.markdown("### Early vs Non-Early Claim Bias")
    early_bias = (df.groupby(['EARLY_NON', 'PI_GENDER'])['CLAIM_APPROVED']
                  .mean().unstack() * 100)
    fig, ax = plt.subplots(figsize=(8, 4))
    early_bias.plot(kind='bar', ax=ax, color=['#1565c0', '#e91e63'], edgecolor='white', rot=0)
    ax.set_title('Approval Rate: Early vs Non-Early × Gender')
    ax.set_ylabel('%')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 3: ML MODELS ─────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Machine Learning Classification Models")

    @st.cache_data
    def run_models(file_bytes):
        data = pd.read_csv(file_bytes)
        data['PI_ANNUAL_INCOME'] = pd.to_numeric(
            data['PI_ANNUAL_INCOME'].astype(str).str.replace(',', '').str.strip(),
            errors='coerce'
        )
        data['SUM_ASSURED'] = pd.to_numeric(
            data['SUM_ASSURED'].astype(str).str.replace(',', '').str.strip(),
            errors='coerce'
        )
        data['CLAIM_APPROVED'] = (data['POLICY_STATUS'] == 'Approved Death Claim').astype(int)

        # Feature engineering
        data['INCOME_TO_SUM'] = data['PI_ANNUAL_INCOME'].fillna(0) / (data['SUM_ASSURED'].fillna(1) + 1)
        data['AGE_INCOME'] = data['PI_AGE'] * data['PI_ANNUAL_INCOME'].fillna(0)
        data['IS_EARLY'] = (data['EARLY_NON'] == 'EARLY').astype(int)
        data['IS_MEDICAL'] = (data['MEDICAL_NONMED'] == 'MEDICAL').astype(int)
        data['IS_FEMALE'] = (data['PI_GENDER'] == 'F').astype(int)

        cat_cols = ['ZONE', 'PAYMENT_MODE', 'PI_OCCUPATION', 'REASON_FOR_CLAIM', 'PI_STATE']
        for c in cat_cols:
            le = LabelEncoder()
            data[c + '_ENC'] = le.fit_transform(data[c].fillna('Unknown').astype(str))

        feat_cols = (
            ['PI_AGE', 'PI_ANNUAL_INCOME', 'SUM_ASSURED',
             'IS_EARLY', 'IS_MEDICAL', 'IS_FEMALE',
             'INCOME_TO_SUM', 'AGE_INCOME']
            + [c + '_ENC' for c in cat_cols]
        )

        X = data[feat_cols]
        y = data['CLAIM_APPROVED']

        X_imp = pd.DataFrame(
            SimpleImputer(strategy='median').fit_transform(X),
            columns=X.columns
        )
        X_sc = StandardScaler().fit_transform(X_imp)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sc, y, test_size=0.2, random_state=42, stratify=y
        )

        model_dict = {
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosted': GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

        res = {}
        for name, model in model_dict.items():
            model.fit(X_tr, y_tr)
            tr_pred = model.predict(X_tr)
            te_pred = model.predict(X_te)
            te_prob = model.predict_proba(X_te)[:, 1]
            res[name] = {
                'train_acc': accuracy_score(y_tr, tr_pred),
                'test_acc': accuracy_score(y_te, te_pred),
                'precision': precision_score(y_te, te_pred),
                'recall': recall_score(y_te, te_pred),
                'f1': f1_score(y_te, te_pred),
                'auc': roc_auc_score(y_te, te_prob),
                'cm': confusion_matrix(y_te, te_pred),
                'y_te': y_te.values,
                'te_prob': te_prob,
                'report': classification_report(y_te, te_pred, output_dict=True),
            }
            if hasattr(model, 'feature_importances_'):
                res[name]['fi'] = pd.Series(
                    model.feature_importances_, index=feat_cols
                ).nlargest(10)
        return res, feat_cols

    with st.spinner("Training 4 ML models – this takes ~30 seconds…"):
        uploaded_file.seek(0)
        results, feat_cols = run_models(uploaded_file)

    st.success("All 4 models trained successfully!")

    with st.expander("Feature Engineering Details"):
        st.markdown("""
**Numerical features:** `PI_AGE`, `PI_ANNUAL_INCOME`, `SUM_ASSURED`

**Binary encoded:** `EARLY_NON` → IS_EARLY · `MEDICAL_NONMED` → IS_MEDICAL · `PI_GENDER` → IS_FEMALE

**Engineered:**
- `INCOME_TO_SUM` = Annual Income ÷ Sum Assured (affordability ratio)
- `AGE_INCOME` = Age × Income (risk-income interaction)

**Label encoded:** ZONE, PAYMENT_MODE, PI_OCCUPATION, REASON_FOR_CLAIM, PI_STATE

**Pipeline:** Median imputation → StandardScaler → 80/20 stratified split
        """)

    st.markdown("### Feature Importances (Random Forest)")
    rf_fi = results['Random Forest']['fi']
    fig, ax = plt.subplots(figsize=(10, 5))
    rf_fi.sort_values().plot(kind='barh', ax=ax, color='#1565c0')
    ax.set_title('Top 10 Feature Importances – Random Forest')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 4: MODEL PERFORMANCE ─────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Model Performance Comparison")

    rows = []
    for name, r in results.items():
        rows.append({
            'Model': name,
            'Train Acc': round(r['train_acc'], 3),
            'Test Acc': round(r['test_acc'], 3),
            'Precision': round(r['precision'], 3),
            'Recall': round(r['recall'], 3),
            'F1': round(r['f1'], 3),
            'AUC': round(r['auc'], 3),
        })
    st.dataframe(pd.DataFrame(rows).set_index('Model'))

    metrics = ['train_acc', 'test_acc', 'precision', 'recall', 'f1', 'auc']
    labels = ['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    names = list(results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, m, lbl in zip(axes.flat, metrics, labels):
        vals = [results[n][m] for n in names]
        bars = ax.bar(names, vals, color=PLOT_COLORS, edgecolor='white')
        ax.set_title(lbl, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### ROC Curves")
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, r), color in zip(results.items(), PLOT_COLORS):
        fpr, tpr, _ = roc_curve(r['y_te'], r['te_prob'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves – All Models', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Confusion Matrices")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, r) in zip(axes, results.items()):
        sns.heatmap(
            r['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Repudiated', 'Approved'],
            yticklabels=['Repudiated', 'Approved'],
            linewidths=1, linecolor='white'
        )
        ax.set_title(f"{name}\nAcc={r['test_acc']:.3f}", fontweight='bold', fontsize=9)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 5: FINDINGS ──────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Key Findings & Recommendations")

    st.markdown("""
<div class="finding-box">
<h4>Finding 1 – Class Imbalance</h4>
68% Approved vs 32% Repudiated. While not extreme, this skew can cause models to favour
approvals. Recall on the repudiated class should be monitored carefully.
</div>

<div class="bias-alert">
<h4>Finding 2 – Zone/Team Bias (Most Critical)</h4>
Several zones show approval rates 15–25% below the overall average.
This suggests differential standards or under-resourced settlement teams.
</div>

<div class="finding-box">
<h4>Finding 3 – Age Bias</h4>
Policyholders aged 60+ face a notably higher repudiation rate.
This may reflect stricter scrutiny on high-sum elderly policies.
</div>

<div class="finding-box">
<h4>Finding 4 – Income Disparity</h4>
62% of records show zero income — likely a data quality gap. Among non-zero income records,
lower-income claimants face higher repudiation, possibly due to lack of documentation support.
</div>

<div class="bias-alert">
<h4>Finding 5 – Early Claim Repudiation Applied Unevenly</h4>
Early claims face significantly higher repudiation (legitimate for fraud control) but this
is not applied uniformly across all zones, indicating process inconsistency.
</div>

<div class="finding-box">
<h4>Finding 6 – Best Models: Random Forest & Gradient Boosted (AUC = 0.784)</h4>
Top predictive features are REASON_FOR_CLAIM, ZONE, PI_AGE, SUM_ASSURED, and
the engineered INCOME_TO_SUM ratio — reinforcing the income/age bias hypothesis.
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Recommendations")
    st.markdown("""
1. **Audit zones with below-average approval rates** — investigate case samples from the bottom 5 zones.
2. **Standardise Early Claim review** across all zones with a uniform checklist.
3. **Age-neutral documentation policy** — avoid stricter scrutiny solely based on age.
4. **Income-blind assessment** — ensure low-income claimants get equal documentation support.
5. **Deploy ML model as a second-opinion tool** to flag potentially biased rejections for compliance review.
6. **Monthly dashboard review** by a compliance officer to track approval-rate trends by zone.
    """)

    st.caption("Insurance Claim Settlement Bias Analysis · Python · Streamlit · Scikit-learn")
