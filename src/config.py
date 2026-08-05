"""
Column groups and the leakage boundary for the MSME stress model.

The single most important thing in this file is LABEL_DEFINING_COLS. The stress
label is derived from delinquency and asset-classification fields, so every
column that defines or mechanically implies the label must be excluded from the
feature set. Keeping that list explicit here — rather than dropping columns by
intuition inside a notebook — is what makes the leakage argument auditable.
"""

ID_COL = "PROSPECTID"

# ── Label definition ────────────────────────────────────────────────
# Stress = the account sits in RBI's SMA-2 bucket or worse.
#
# Under RBI's Income Recognition and Asset Classification (IRAC) norms an
# account is tracked as SMA-0 (1-30 days overdue), SMA-1 (31-60), SMA-2 (61-90)
# and NPA (90+). SMA-2 is the supervisory threshold at which Indian banks are
# required to report and act on a stressed account, which makes it the natural
# operational definition of "stressed" for an early-warning system.
#
# We mark a borrower stressed if either:
#   - they have ever been 60+ days past due (SMA-2 or worse), or
#   - they hold any trade line classified sub-standard, doubtful or loss (NPA).
DPD_60_COL = "num_times_60p_dpd"
NPA_COUNT_COLS = ["num_sub", "num_dbt", "num_lss"]

LABEL_COL = "stress_label"

# ── Leakage boundary ────────────────────────────────────────────────
# 1. Asset classification counts. These ARE the label for the NPA half.
#    num_std is included too: standard-asset counts are produced by the same
#    classification process and reveal the classification outcome.
ASSET_CLASS_COLS = [
    "num_std", "num_std_6mts", "num_std_12mts",
    "num_sub", "num_sub_6mts", "num_sub_12mts",
    "num_dbt", "num_dbt_6mts", "num_dbt_12mts",
    "num_lss", "num_lss_6mts", "num_lss_12mts",
]

# 2. Delinquency and days-past-due history. num_times_60p_dpd is half the label
#    outright; the rest are the same underlying event counted a different way
#    (max_delinquency_level correlates 0.54 with the label, num_times_delinquent
#    0.57). A model given these is reading the answer, not predicting it.
DELINQUENCY_COLS = [
    "num_times_delinquent", "num_times_30p_dpd", "num_times_60p_dpd",
    "max_delinquency_level", "max_recent_level_of_deliq", "recent_level_of_deliq",
    "num_deliq_6mts", "num_deliq_12mts", "num_deliq_6_12mts",
    "max_deliq_6mts", "max_deliq_12mts",
    "time_since_first_deliquency", "time_since_recent_deliquency",
]

# 3. Payment-behaviour fields. Missed payments and time-since-last-payment are
#    the mechanical precursors of a DPD count — the same phenomenon observed one
#    step earlier. Excluding them keeps the claim clean: this model predicts
#    payment distress WITHOUT using payment history.
PAYMENT_BEHAVIOUR_COLS = ["Tot_Missed_Pmnt", "time_since_recent_payment"]

# 4. Bureau/underwriting outputs. Credit_Score and Approved_Flag are downstream
#    of the same delinquency record the label comes from.
BUREAU_OUTPUT_COLS = ["Credit_Score", "Approved_Flag"]

LABEL_DEFINING_COLS = (
    ASSET_CLASS_COLS
    + DELINQUENCY_COLS
    + PAYMENT_BEHAVIOUR_COLS
    + BUREAU_OUTPUT_COLS
)

# ── Permitted feature families (the "leading indicator" set) ────────
# What survives the boundary is credit-seeking behaviour, balance-sheet
# utilisation, portfolio structure and borrower profile. This is exactly the
# hypothesis the project set out to test: that these signals flag stress before
# repayment behaviour deteriorates.

ENQUIRY_COLS = [
    "tot_enq", "CC_enq", "CC_enq_L6m", "CC_enq_L12m",
    "PL_enq", "PL_enq_L6m", "PL_enq_L12m",
    "time_since_recent_enq", "enq_L12m", "enq_L6m", "enq_L3m",
    "pct_PL_enq_L6m_of_L12m", "pct_CC_enq_L6m_of_L12m",
    "pct_PL_enq_L6m_of_ever", "pct_CC_enq_L6m_of_ever",
]

UTILISATION_COLS = [
    "CC_utilization", "PL_utilization",
    "max_unsec_exposure_inPct", "pct_currentBal_all_TL",
]

PORTFOLIO_COLS = [
    "Total_TL", "Tot_Closed_TL", "Tot_Active_TL",
    "Total_TL_opened_L6M", "Tot_TL_closed_L6M",
    "Total_TL_opened_L12M", "Tot_TL_closed_L12M",
    "pct_tl_open_L6M", "pct_tl_closed_L6M",
    "pct_tl_open_L12M", "pct_tl_closed_L12M",
    "pct_active_tl", "pct_closed_tl",
    "pct_of_active_TLs_ever", "pct_opened_TLs_L6m_of_L12m",
    "Auto_TL", "CC_TL", "Consumer_TL", "Gold_TL", "Home_TL", "PL_TL",
    "Secured_TL", "Unsecured_TL", "Other_TL",
    "Age_Oldest_TL", "Age_Newest_TL",
    "CC_Flag", "PL_Flag", "HL_Flag", "GL_Flag",
]

PROFILE_COLS = [
    "AGE", "NETMONTHLYINCOME", "Time_With_Curr_Empr",
]

CATEGORICAL_COLS = [
    "MARITALSTATUS", "EDUCATION", "GENDER",
    "last_prod_enq2", "first_prod_enq2",
]

RAW_NUMERIC_FEATURES = (
    ENQUIRY_COLS + UTILISATION_COLS + PORTFOLIO_COLS + PROFILE_COLS
)

# ── Missing-data handling ───────────────────────────────────────────
# The source data encodes "not reported" as -99999 / -99998. Blind median fill
# destroys real signal here: CC_utilization is 92.8% sentinel and PL_utilization
# 86.6%, so median-filling turns them into near-constants. We fill AND emit a
# binary indicator, because "no credit-card utilisation on record" is itself
# informative (it usually means a thin file, not an average one).
SENTINEL_VALUES = [-99999, -99998]

# Emit a _missing indicator for any column whose sentinel rate exceeds this.
MISSING_INDICATOR_THRESHOLD = 0.01

# Ordinal encoding for education — the one categorical with a natural order.
EDUCATION_ORDER = {
    "OTHERS": 0,
    "SSC": 1,
    "12TH": 2,
    "UNDER GRADUATE": 3,
    "GRADUATE": 4,
    "POST-GRADUATE": 5,
    "PROFESSIONAL": 6,
}

# ── Paths ───────────────────────────────────────────────────────────
# Resolved against the project root rather than the current directory, so the
# training script and the dashboard work regardless of where they are launched
# from.
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parent.parent

MERGED_DATA = str(PROJECT_ROOT / "data/processed/credit_merged_clean.csv")
FEATURE_TABLE = str(PROJECT_ROOT / "data/processed/features_v2.csv")
FEATURE_PARAMS = str(PROJECT_ROOT / "models/feature_params.json")
MODEL_PATH = str(PROJECT_ROOT / "models/stress_model_v2.pkl")
METRICS_PATH = str(PROJECT_ROOT / "models/metrics_v2.json")
SCORED_PORTFOLIO = str(PROJECT_ROOT / "data/processed/scored_portfolio_v2.csv")
