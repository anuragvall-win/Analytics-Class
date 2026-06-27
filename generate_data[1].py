import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

SEG_LABELS = [
    "Ultra-Premium Independent",
    "Heritage Luxury Brand",
    "Boutique Artisanal Atelier",
    "Mid-Tier Commercial Brand",
    "OEM / Contract Manufacturer",
]

def generate_dataset(n=1100, seed=42):
    np.random.seed(seed)
    N = n
    SEG_PROPS = [0.22, 0.28, 0.30, 0.15, 0.05]
    seg_sizes = [int(p * N) for p in SEG_PROPS]
    seg_sizes[-1] = N - sum(seg_sizes[:-1])
    segment_ids = np.concatenate([np.full(s, i) for i, s in enumerate(seg_sizes)])
    np.random.shuffle(segment_ids)

    budget_mean  = np.array([2_800_000, 16_000_000,  380_000, 4_200_000, 8_500_000])
    budget_sd    = np.array([  700_000,  3_500_000,   90_000,   900_000, 2_000_000])
    emp_mu       = np.array([16, 820, 10, 210, 600])
    years_mu     = np.array([42, 88, 28, 20, 14])
    years_sd     = np.array([10, 20, 10,  7,  5])
    suppliers_mu = np.array([ 7, 18,  4, 12, 25])
    suppliers_sd = np.array([ 2,  4,  1,  3,  6])
    otd_mu       = np.array([94.5, 91.5, 96.5, 86.5, 89.5])
    otd_sd       = np.array([ 1.5,  2.0,  1.2,  3.0,  2.5])
    rejection_mu = np.array([ 0.8,  1.3,  0.5,  3.8,  4.0])
    rejection_sd = np.array([ 0.2,  0.3,  0.2,  0.8,  1.0])
    audit_mu     = np.array([ 3.8,  3.0,  4.2,  1.4,  1.2])
    audit_sd     = np.array([ 0.7,  0.6,  0.8,  0.5,  0.4])
    inv_turn_mu  = np.array([ 2.0,  3.5,  1.4,  5.2,  7.0])
    inv_turn_sd  = np.array([ 0.4,  0.7,  0.3,  1.0,  1.4])
    sil_use_mu   = np.array([18.0, 12.0, 22.0,  5.5,  1.5])
    sil_use_sd   = np.array([ 5.0,  4.0,  6.0,  2.5,  1.0])
    sil_int_mu   = np.array([ 6.2,  5.8,  6.5,  4.1,  2.4])
    sil_int_sd   = np.array([ 0.5,  0.6,  0.5,  0.8,  0.9])
    sil_price_mu = np.array([1250, 680, 1480, 320, 90])
    sil_price_sd = np.array([ 180, 100,  200,  60, 20])
    bc_will_mu   = np.array([ 5.8,  5.5,  6.1,  3.9,  2.7])
    bc_will_sd   = np.array([ 0.6,  0.7,  0.5,  0.9,  1.0])
    nft_int_mu   = np.array([ 5.5,  5.2,  5.9,  3.5,  2.2])
    nft_int_sd   = np.array([ 0.7,  0.8,  0.6,  1.0,  1.1])
    wtp_mu       = np.array([22.0, 18.0, 25.0, 10.0,  3.5])
    wtp_sd       = np.array([ 4.0,  4.0,  4.5,  3.5,  1.5])
    dig_mu       = np.array([ 5.5,  5.8,  4.8,  4.2,  3.5])
    dig_sd       = np.array([ 0.8,  0.7,  0.9,  0.9,  1.0])
    codesign_mu  = np.array([ 6.3,  5.6,  6.7,  4.0,  2.5])
    codesign_sd  = np.array([ 0.5,  0.7,  0.5,  0.9,  1.0])
    likert_mu = np.array([
        [6.8, 6.6, 6.7, 6.7, 5.9, 2.1, 6.5],
        [6.5, 6.4, 6.2, 6.0, 6.1, 3.0, 6.3],
        [6.9, 6.8, 6.8, 6.9, 6.6, 2.3, 6.8],
        [5.1, 4.8, 4.5, 4.2, 4.6, 5.4, 4.8],
        [4.0, 3.9, 3.2, 3.1, 3.7, 6.6, 4.0],
    ])
    pipeline_probs = [
        [0.15, 0.30, 0.35, 0.20],
        [0.10, 0.25, 0.38, 0.27],
        [0.20, 0.30, 0.32, 0.18],
        [0.30, 0.38, 0.22, 0.10],
        [0.55, 0.30, 0.10, 0.05],
    ]
    batch_probs = [
        [0.50, 0.35, 0.12, 0.02, 0.01],
        [0.25, 0.45, 0.22, 0.07, 0.01],
        [0.60, 0.32, 0.07, 0.01, 0.00],
        [0.08, 0.20, 0.42, 0.25, 0.05],
        [0.01, 0.02, 0.07, 0.30, 0.60],
    ]
    custom_probs = [
        [0.02, 0.12, 0.38, 0.48],
        [0.03, 0.18, 0.42, 0.37],
        [0.01, 0.08, 0.30, 0.61],
        [0.15, 0.38, 0.32, 0.15],
        [0.55, 0.32, 0.10, 0.03],
    ]
    trace_probs = [
        [0.10, 0.15, 0.50, 0.20, 0.05],
        [0.05, 0.10, 0.60, 0.22, 0.03],
        [0.20, 0.30, 0.35, 0.12, 0.03],
        [0.08, 0.20, 0.58, 0.08, 0.06],
        [0.05, 0.12, 0.72, 0.04, 0.07],
    ]
    bc_fam_probs = [
        [0.05, 0.20, 0.38, 0.28, 0.09],
        [0.04, 0.18, 0.40, 0.28, 0.10],
        [0.07, 0.22, 0.36, 0.25, 0.10],
        [0.18, 0.35, 0.30, 0.12, 0.05],
        [0.38, 0.38, 0.17, 0.05, 0.02],
    ]
    cert_probs = [
        [0.02, 0.08, 0.38, 0.52],
        [0.01, 0.05, 0.28, 0.66],
        [0.02, 0.08, 0.42, 0.48],
        [0.05, 0.42, 0.25, 0.28],
        [0.12, 0.55, 0.10, 0.23],
    ]
    pain_probs = [
        [0.35, 0.30, 0.12, 0.18, 0.05],
        [0.32, 0.28, 0.18, 0.15, 0.07],
        [0.38, 0.32, 0.10, 0.16, 0.04],
        [0.18, 0.22, 0.28, 0.20, 0.12],
        [0.08, 0.15, 0.32, 0.10, 0.35],
    ]
    lead_time_mu = [25, 30, 20, 45, 60]
    lead_time_sd = [ 5,  6,  4,  8, 10]

    rows = []
    for i, s in enumerate(segment_ids):
        budget    = float(np.random.lognormal(np.log(budget_mean[s]), budget_sd[s]/budget_mean[s]))
        emps      = max(1, int(np.random.lognormal(np.log(emp_mu[s]), 0.4)))
        years_b   = max(1, int(np.random.normal(years_mu[s], years_sd[s])))
        num_sup   = max(1, int(np.random.normal(suppliers_mu[s], suppliers_sd[s])))
        otd       = min(99.9, max(50.0, round(float(np.random.normal(otd_mu[s], otd_sd[s])), 1)))
        rejection = max(0.1, round(float(np.random.normal(rejection_mu[s], rejection_sd[s])), 2))
        audit_f   = max(0.5, round(float(np.random.normal(audit_mu[s], audit_sd[s])), 1))
        inv_turn  = max(0.5, round(float(np.random.normal(inv_turn_mu[s], inv_turn_sd[s])), 2))
        sil_use   = min(100.0, max(0.0, round(float(np.random.normal(sil_use_mu[s], sil_use_sd[s])), 1)))
        sil_int   = int(np.clip(round(np.random.normal(sil_int_mu[s], sil_int_sd[s])), 1, 7))
        sil_price = max(30.0, round(float(np.random.normal(sil_price_mu[s], sil_price_sd[s])), 2))
        bc_will   = int(np.clip(round(np.random.normal(bc_will_mu[s], bc_will_sd[s])), 1, 7))
        nft_int   = int(np.clip(round(np.random.normal(nft_int_mu[s], nft_int_sd[s])), 1, 7))
        wtp       = max(0.0, round(float(np.random.normal(wtp_mu[s], wtp_sd[s])), 1))
        dig_r     = int(np.clip(round(np.random.normal(dig_mu[s], dig_sd[s])), 1, 7))
        codesign  = int(np.clip(round(np.random.normal(codesign_mu[s], codesign_sd[s])), 1, 7))
        lk        = np.clip(np.round(np.random.normal(likert_mu[s], 0.6)).astype(int), 1, 7)
        stage_idx = np.random.choice(4, p=pipeline_probs[s])
        stage_nm  = ["Awareness","Consideration","Intent","Decision"][stage_idx]
        base_int  = stage_idx * 2.0 + lk[0]*0.25 + sil_int*0.2
        intent    = int(np.clip(round(base_int + np.random.normal(0,0.7)), 1, 10))
        deal_val  = max(500.0, round(budget*(0.02+stage_idx*0.035+np.random.normal(0,0.01)), 2))
        contacted = np.random.rand() < [0.22,0.42,0.68,0.90][stage_idx]
        sw_base   = 5.5 - lk[6]*0.35 - min(years_b/30,1.5) + np.random.normal(0,0.5)
        switch    = int(np.clip(round(sw_base), 1, 7))
        lt_custom = max(3, int(np.random.normal(lead_time_mu[s], lead_time_sd[s])))

        rows.append({
            "respondent_id":                           f"RESP-{i+1:04d}",
            "segment":                                 SEG_LABELS[s],
            "segment_id":                              int(s),
            "years_in_business":                       years_b,
            "num_employees":                           emps,
            "annual_procurement_budget_usd":           round(budget, 2),
            "num_active_suppliers":                    num_sup,
            "on_time_delivery_rate_pct":               otd,
            "quality_rejection_rate_pct":              rejection,
            "supplier_audit_frequency_per_year":       audit_f,
            "inventory_turnover_ratio":                inv_turn,
            "certification_level":                     np.random.choice(["None","ISO9001","COSC","Multiple"], p=cert_probs[s]),
            "current_silicon_usage_pct":               sil_use,
            "interest_in_silicon_adoption":            sil_int,
            "silicon_component_price_usd":             sil_price,
            "preferred_batch_size":                    np.random.choice(["1-10","11-50","51-200","201-1000","1000+"], p=batch_probs[s]),
            "customisation_frequency":                 np.random.choice(["Never","Occasionally","Regularly","Always"], p=custom_probs[s]),
            "acceptable_lead_time_custom_days":        lt_custom,
            "blockchain_familiarity":                  np.random.choice(["None","Aware","Exploring","Piloting","Implemented"], p=bc_fam_probs[s]),
            "blockchain_adoption_willingness":         bc_will,
            "nft_provenance_certificate_interest":     nft_int,
            "traceability_method_current":             np.random.choice(["Paper_Records","Spreadsheets","ERP_System","Partial_Blockchain","None"], p=trace_probs[s]),
            "willingness_to_pay_traceability_premium_pct": wtp,
            "digital_supply_chain_readiness":          dig_r,
            "codesign_interest":                       codesign,
            "importance_quality":                      int(lk[0]),
            "importance_traceability":                 int(lk[1]),
            "importance_customisation":                int(lk[2]),
            "importance_innovation":                   int(lk[3]),
            "importance_sustainability":               int(lk[4]),
            "importance_cost":                         int(lk[5]),
            "importance_supplier_relationship":        int(lk[6]),
            "lead_stage":                              stage_nm,
            "purchase_intent_score":                   intent,
            "estimated_deal_value_usd":                deal_val,
            "contacted_supplier_last_6mo":             "Yes" if contacted else "No",
            "likelihood_to_switch_supplier":           switch,
            "primary_pain_point":                      np.random.choice(
                ["Traceability_Gaps","Quality_Inconsistency","Long_Lead_Times","No_Silicon_Option","High_Minimum_Orders"],
                p=pain_probs[s]),
        })

    df = pd.DataFrame(rows)

    # Outliers
    out_cols = ["annual_procurement_budget_usd","silicon_component_price_usd",
                "quality_rejection_rate_pct","willingness_to_pay_traceability_premium_pct"]
    for idx in np.random.choice(df.index, size=int(0.035*N), replace=False):
        col = np.random.choice(out_cols)
        df.loc[idx, col] = abs(float(df[col].mean() + np.random.choice([-1,1])*df[col].std()*np.random.uniform(3.2,5.5)))

    # Missing values
    for col in ["current_silicon_usage_pct","willingness_to_pay_traceability_premium_pct",
                "supplier_audit_frequency_per_year","inventory_turnover_ratio",
                "nft_provenance_certificate_interest"]:
        df.loc[np.random.rand(N) < 0.016, col] = np.nan

    # Right-skew amplification
    df["annual_procurement_budget_usd"] = (
        df["annual_procurement_budget_usd"].astype(float) * (np.random.chisquare(df=3, size=N)/3)
    )
    return df.reset_index(drop=True)

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("silicon_watch_survey_RAW.csv", index=False)
    print(f"Saved {df.shape[0]}x{df.shape[1]}")
    print(df["segment"].value_counts().to_string())
