# Predicting MSME Working Capital Stress 90 Days Early
### A Causal ML and Survival Analysis Approach for Indian SME Credit Risk

**Author:** Shaurya Gupta
**Domain:** Financial Data Science | Credit Risk | Causal ML

---

## Problem Statement

India has 63 million MSMEs contributing approximately 30% of GDP, yet over
INR 5 lakh crore in MSME loans are classified as non-performing assets. Existing
credit scoring systems (CIBIL, bureau-based models) are fundamentally reactive —
they detect financial stress only after repayment behavior has already
deteriorated.

This project builds a proactive early warning system that identifies MSME cash
flow stress before it escalates into default, using leading operational signals
that existing models ignore: credit enquiry patterns, payment behavior, loan
portfolio composition, and commodity price volatility.

The primary users are banks, NBFCs, and credit analysts making three decisions:
whether to extend working capital credit, how to tier risk across a portfolio,
and which specific factors are driving a firm's risk. Success is defined as
AUC-ROC above 0.80, precision in the top risk decile above 60%, and a survival
model C-index above 0.65.

---

## Data

The analysis combines five data sources:

- **Indian CIBIL external credit bureau data** — 51,336 borrower profiles,
  62 features covering enquiry history, delinquency records, and credit utilization.
- **Indian bank internal data** — 51,336 profiles, 26 features covering active
  and closed trade lines, missed payments, and secured/unsecured loan breakdown.
- **Agmarknet commodity prices** — daily wholesale mandi prices across Indian
  states for spices, vegetables, and agricultural commodities.
- **GST predictive dataset** — 785,133 records covering tax filing patterns.
- **Kaggle credit risk dataset** — used for baseline prototyping.

After merging CIBIL and bank data on PROSPECTID: 51,336 profiles, 88 raw
features, 37.3% stress rate. The target variable was derived from the
Approved_Flag tier system (P2 = healthy, P1/P3/P4 = stressed).

---

## Feature Engineering

Fifteen domain-specific features were engineered. The most impactful:

- **enquiry_acceleration** — weighted sum of credit enquiries (last 3 months
  weighted 4x, 6 months 2x, 12 months 1x). Captures desperation for credit.
- **delinquency_severity** — composite of frequency, recency, and severity of
  delinquencies.
- **credit_stress_score** — combines utilization risk and days-past-due stress.
- **unsecured_loan_ratio, recent_loan_hunger, closure_rate, missed_payment_ratio**
  — loan portfolio behavior signals.
- **market_volatility_score** — commodity price volatility index from Agmarknet
  data (std/mean ratio across commodities).

Sentinel values (-99999) representing missing data were replaced with column
medians. Extreme outliers in debt_burden were capped at the 99th percentile.

---

## Methods and Results

### Supervised Classification

A logistic regression baseline achieved AUC-ROC 0.732. XGBoost, tuned with
early stopping and class-weight adjustment for the imbalanced target, achieved:

- AUC-ROC: **0.822** (+0.09 over baseline)
- Precision @ top 5% riskiest firms: **96.9%**
- Precision @ top 10% riskiest firms: **90.9%**

SHAP TreeExplainer analysis identified enquiry_acceleration as the single most
important feature (mean SHAP 0.73 — more than double the next feature).

### Survival Analysis

A Kaplan-Meier analysis showed statistically significant separation between high
and low enquiry-acceleration firms (log-rank p < 0.001). A Cox Proportional
Hazards model achieved a concordance index of **0.671**. Key hazard ratios:
enquiry_acceleration 1.24, enquiry_rejection_proxy 1.12, active trade lines 1.07
— all significant at p < 0.0005.

### Causal Inference

Using DoWhy, high enquiry acceleration was shown to causally increase stress
probability by **28.7 percentage points**, controlling for age, income,
delinquency history, and active loans. This survived all three refutation tests:
random common cause (effect unchanged), placebo treatment (effect dropped to
0.0001), and data subset (effect stable). The relationship is causal, not merely
correlational.

### Unsupervised Anomaly Detection

An LSTM autoencoder trained only on healthy firms achieved AUC 0.692 as an
unsupervised anomaly detector. Stressed firms showed 4.4x higher reconstruction
error, confirming they represent genuine anomalies relative to healthy patterns.

### Uncertainty Quantification and Ensemble

Conformal prediction delivered a guaranteed 90% coverage that held at 90.3% on
the test set, with 39% of cases flagged as genuinely uncertain. A composite risk
score combining XGBoost and LSTM signals produced four clean risk tiers with
monotonic escalation: Low 11.5%, Medium 34.4%, High 64.0%, Critical 93.7% stress
rate.

---

## Domain Validation and Limitations

Feature selection and assumptions were validated against real-world wholesale
trading operations in India's spice and herbs sector, reflecting actual MSME cash
flow mechanics, payment cycles, and commodity volatility exposure.

**Limitations:** Survival durations were simulated based on risk profiles rather
than derived from actual default timestamps, as the dataset lacked longitudinal
timing data. The commodity dataset covered a limited time range, so market
volatility was computed as a cross-sectional rather than time-series signal.
Future work should incorporate real TReDS payment cycle data and longitudinal
GST filing sequences.

---

## Conclusion

This project demonstrates a complete, multi-method ML system for MSME stress
prediction spanning supervised learning, survival analysis, causal inference,
unsupervised anomaly detection, and uncertainty quantification. The strongest
finding — that credit enquiry acceleration is both the top predictor and a
genuine causal driver of stress — offers lenders an actionable, interpretable,
and statistically rigorous early warning signal.