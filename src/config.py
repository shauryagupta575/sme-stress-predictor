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
# Stress = the borrower deteriorated inside a FIXED 12-month observation window.
#
# Under RBI's Income Recognition and Asset Classification (IRAC) norms an
# account is tracked as SMA-0 (1-30 days overdue), SMA-1 (31-60), SMA-2 (61-90)
# and NPA (90+). The SMA framework is a rolling current-state view — it is what
# supervisors and lenders actually monitor month to month.
#
# We mark a borrower stressed if, within the last 12 months, either:
#   - any trade line went delinquent, or
#   - any trade line was classified sub-standard, doubtful or loss (NPA).
#
# WHY A FIXED WINDOW — this is the important design decision.
# An earlier version used a lifetime definition ("ever 60+ days past due").
# That is not exposure-comparable: a borrower holding 20 accounts over 15 years
# has mechanically more opportunities to have slipped once than a borrower with
# 2 accounts over 3 years, so the label partly measured how much credit history
# someone had rather than how much distress. It showed up in the data:
#
#   label prevalence, 11+ accounts vs 1-2 accounts   exposure-only-model AUC
#   lifetime label          5.45x                            0.782
#   fixed 12-month window   2.39x                            0.698
#
# The "exposure-only" figure is the AUC reachable using nothing but account
# counts and credit-file ages — a floor that any honest evaluation has to beat.
# Moving to a fixed window cut it by 0.084 and more than halved the prevalence
# gradient. The residual 2.39x is not assumed to be artefact: holding many
# credit lines is plausibly correlated with real risk. It is reported rather
# than explained away, and metrics are also broken out within exposure bands.
DELIQ_12M_COL = "num_deliq_12mts"
NPA_12M_COLS = ["num_sub_12mts", "num_dbt_12mts", "num_lss_12mts"]

LABEL_COL = "stress_label"

# Account-count and credit-age fields, used to build the exposure-only baseline
# model and the exposure strata. These stay available as ordinary features — the
# point is to measure how far the full model beats them, not to ban them.
EXPOSURE_COLS = [
    "Total_TL", "Tot_Active_TL", "Tot_Closed_TL",
    "Age_Oldest_TL", "Age_Newest_TL",
]
EXPOSURE_BANDS = [0, 2, 5, 10, 10_000]
EXPOSURE_BAND_LABELS = ["1-2", "3-5", "6-10", "11+"]

# ── Window overlap: the direction-of-the-arrow problem ──────────────
# The label covers the last 12 months. So do many of the features. A borrower who
# went delinquent in month 3 may then have made a burst of enquiries and opened
# accounts in months 4-12 precisely BECAUSE they were in distress — in which case
# `enq_L3m` is a consequence of the outcome, not a signal of it. The source is a
# single cross-sectional snapshot with no as-of dates, so the ordering cannot be
# recovered from the data and the direction is formally unidentifiable.
#
# What can be done is bound the exposure to it. These are the features whose
# measurement window overlaps the label window; retraining without them gives a
# model that cannot be reading post-outcome behaviour, and the gap between the two
# is the honest size of the concern. See train.py::no_overlap_variant.
#
# Data-availability indicators (`*_missing`) are deliberately NOT listed: they
# describe whether a field was ever reported, not activity inside the window.
WINDOW_OVERLAP_FEATURES = [
    # engineered from 3/6/12-month activity counts
    "enquiry_acceleration", "enquiry_recency_share", "credit_hunger_ratio",
    "enquiry_freshness", "recent_loan_hunger", "closure_rate",
    "open_close_imbalance",
    # depend on the newest account's age, which may fall inside the window
    "portfolio_age_span", "newest_tl_age",
    # raw windowed counts carried through
    "enq_L3m", "enq_L6m", "enq_L12m", "pct_tl_open_L6M", "pct_tl_closed_L6M",
]

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

# 2. Delinquency and days-past-due history. num_deliq_12mts IS the label; the
#    rest are the same underlying event counted a different way or over a
#    different window. Lifetime measures such as num_times_60p_dpd would be
#    defensible predictors of a 12-month-window label, but they are excluded
#    anyway so the claim can stay absolute: no delinquency or payment-behaviour
#    field of any kind reaches the model.
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
FEATURE_PARAMS = str(PROJECT_ROOT / "models/feature_params.json")
MODEL_PATH = str(PROJECT_ROOT / "models/stress_model_v2.pkl")
METRICS_PATH = str(PROJECT_ROOT / "models/metrics_v2.json")
SCORED_PORTFOLIO = str(PROJECT_ROOT / "data/processed/scored_portfolio_v2.csv")
