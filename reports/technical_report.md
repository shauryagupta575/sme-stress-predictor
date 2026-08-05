# Detecting MSME Credit Stress Without Payment History

### An exposure-controlled triage model on Indian credit bureau data

**Author:** Shaurya Gupta
**Domain:** Financial Data Science | Credit Risk | Applied ML

---

## Summary

This project builds a model that ranks Indian MSME-proprietor borrowers by whether
their credit deteriorated during the last 12 months, using credit-seeking
behaviour, balance-sheet utilisation, portfolio structure and borrower profile.
Every delinquency, payment-history and asset-classification field is excluded by
rule, so the model never reads the record that defines the outcome.

It is a **detection and triage** model, not a forecast. The label looks backwards
over a fixed window; the data is one cross-sectional snapshot with no as-of dates;
and feature windows overlap the label window, so for a measurable share of the
signal the direction of association cannot be established. That share is
quantified in "Direction of the arrow" rather than left as a caveat. The title of
an earlier version of this report began "Predicting… 90 Days Early"; every part of
that framing has been withdrawn.

On a held-out set of 10,268 borrowers:

| Metric | Cross-validated (5-fold) | Held-out test |
|---|---|---|
| ROC-AUC | 0.7655 ± 0.0045 | 0.7744 |
| Average precision | 0.3944 ± 0.0150 | 0.4058 |
| Precision @ top 5% | 52.7% ± 2.9% | 55.8% |
| Precision @ top 10% | 46.0% ± 1.5% | 47.5% |
| Recall @ top 10% | 27.4% ± 0.9% | 28.3% |

Base stress rate is 16.76%, so average precision reads against a 0.168 baseline
rather than 0.5, and precision at the top decile is a 2.83× lift over random
review.

The result the project rests on is not the AUC. It is that the model beats an
**exposure-only baseline** — one given nothing but account counts and credit-file
ages — by **+0.0713 AUC** (0.7744 against 0.7031), and holds a narrow band of
0.736–0.768 *within* every exposure stratum. Reaching that comparison required
discarding two earlier versions of this project's headline claims, both of which
reported higher numbers.

---

## Problem

India has roughly 63 million MSMEs contributing about 30% of GDP, alongside a
large stock of MSME loans classified as non-performing. Bureau-based scoring is
largely reactive: it detects stress after repayment behaviour has already
deteriorated. The question here is whether deterioration can be ranked ahead of
time from signals that are *not* repayment behaviour.

The intended users are credit teams at banks and NBFCs making one operational
decision: given limited review capacity, which accounts should a human look at
first.

---

## Data

- **Indian credit bureau extract** — 51,336 borrower profiles, 62 fields covering
  enquiry history, delinquency records, asset classification and utilisation.
- **Bank internal extract** — 51,336 profiles, 26 fields covering active and
  closed trade lines, product mix and secured/unsecured composition.

Joined on `PROSPECTID`: 51,336 rows, 88 raw columns.

Two further sources were explored and are **not** used by the final model.
Agmarknet commodity prices were intended to supply an input-cost volatility
signal, but the extract covers too short a window to build a time series; a
cross-sectional volatility index was tried and collapsed to a single constant
across all rows, contributing nothing. A GST filing dataset (785,133 records) has
no reliable join key to the bureau records. Both are recorded here rather than
quietly dropped.

These are retail credit-bureau records used as a proxy for MSME proprietor risk.
They are not firm-level financials. That is the largest limitation of the work.

---

## The label, and why it was rebuilt twice

The label is the most consequential decision in this project, and the first two
attempts were wrong. Both failures are documented because the diagnostics that
caught them are the substance of the work.

### Attempt 1 — derived from the approval tier (discarded)

The original label was `Approved_Flag != "P2"`: tier P2 healthy, P1/P3/P4
stressed. P1 is conventionally the *best* tier, so this merged the best and worst
borrowers into one positive class. It produced an AUC of 0.825 — which is the
problem. The number looked fine while the target was close to meaningless.

The diagnostic that settled it: measured against a behaviourally-defined stress
label, the approval tiers barely separate — 8.3% stress in P2 against 13.0% in P1
and 17.9% in P4. `Approved_Flag` does not encode credit stress.

### Attempt 2 — lifetime delinquency (discarded)

The replacement was regulator-grounded: stress means the borrower has *ever* been
60+ days past due, or holds an NPA-classified trade line. Under RBI's asset
classification norms, 61–90 days overdue is the SMA-2 bucket — the supervisory
threshold at which lenders must report and act. Prevalence 10.63%, AUC 0.8135.

This label is **not exposure-comparable**, which turned out to matter more than
anything else in the project. Because the label counts *events*, a borrower
holding 20 accounts across 15 years has mechanically more opportunities to have
slipped once than a borrower with 2 accounts across 3 years. The label was partly
measuring how much credit history someone had.

| Diagnostic | Lifetime label |
|---|---|
| Prevalence, 1–2 trade lines | 4.8% |
| Prevalence, 11+ trade lines | 26.4% |
| Prevalence gradient (highest / lowest band) | **5.45×** |
| AUC from account counts and file ages alone | **0.782** |
| Full 57-feature model AUC | 0.8135 |
| **Margin over pure exposure** | **+0.025** |

A six-feature model doing little more than counting trade lines reached 0.782.
The full model's entire contribution above that was 0.025 AUC. Reported as
"AUC 0.81", that model was mostly an account counter.

### Final label — a fixed 12-month observation window

Stress means: **within the last 12 months, any trade line went delinquent, or any
trade line was classified sub-standard, doubtful or loss.**

The SMA framework is a rolling current-state view — what supervisors and lenders
monitor month to month — so a fixed recent window is the right shape, not a
compromise. Every borrower is observed over the same period.

| Diagnostic | Lifetime | Fixed 12-month |
|---|---|---|
| Prevalence | 10.63% | 16.76% |
| Prevalence gradient across exposure bands | 5.45× | **2.39×** |
| Exposure-only baseline AUC | 0.782 | **0.7031** |
| Full model AUC | 0.8135 | 0.7744 |
| **Margin over exposure** | +0.025 | **+0.0713** |

The headline AUC fell by 0.039 and the model became substantially more
defensible. The margin over the exposure baseline nearly tripled.

The residual 2.39× gradient is **not** assumed to be artefact — holding many
credit lines is plausibly correlated with genuine risk. It is reported and
controlled for in evaluation rather than adjusted away.

---

## Leakage controls

29 columns that define or mechanically imply the label are excluded by an
explicit rule in `src/config.py`, in four groups:

1. **Asset classification counts** (12) — `num_std`/`num_sub`/`num_dbt`/`num_lss`
   and their 6- and 12-month variants. These *are* the label's NPA arm.
2. **Delinquency and days-past-due history** (13) — `num_deliq_12mts` is the label
   itself; the rest are the same event counted differently or over a different
   window.
3. **Payment behaviour** (2) — missed payments and time-since-last-payment are the
   mechanical precursors of a DPD count.
4. **Bureau/underwriting outputs** (2) — `Credit_Score` and `Approved_Flag` sit
   downstream of the same delinquency record.

Lifetime measures such as `num_times_60p_dpd` would be defensible predictors of a
12-month-window label. They are excluded anyway so the claim can stay absolute:
**no delinquency or payment-behaviour field of any kind reaches the model.**

The exclusion is asserted in code (`features.assert_no_leakage`) and the training
run raises if a banned column reaches the feature matrix. A leakage bug that
surfaces as a suspiciously good AUC is far harder to catch than one that crashes.

---

## Features

57 features across four permitted families. Feature statistics — per-column
medians, clip bounds, normalisers — are fitted on the training split only and
persisted to `models/feature_params.json`, so the training pipeline and the
dashboard produce identical features for the same borrower. The earlier notebook
implementation recomputed these from whatever data was to hand, which would have
caused silent train/serve skew.

**Credit-seeking behaviour.** `enquiry_acceleration` (enquiries weighted 4× at 3
months, 2× at 6, 1× at 12), enquiries per active trade line, share of lifetime
enquiries in the last 6 months, unsecured-product enquiry share, enquiry recency.

**Balance-sheet utilisation.** Credit-card and personal-loan utilisation,
unsecured exposure relative to income, current-balance pressure.

**Portfolio structure.** Unsecured share, recent account opening weighted toward
6 months, closure rate, open/close imbalance, credit-file age span, product-mix
concentration.

**Borrower profile.** Age, log income, employment tenure, education (ordinal),
plus low-cardinality categoricals.

### Missing data

`CC_utilization` is 92.8% sentinel-coded (`-99999` = not reported) and
`PL_utilization` is 86.6%. The original pipeline median-filled these, which turns
a mostly-absent column into a near-constant. Here each is filled *and* paired
with an explicit missingness indicator, on the reasoning that "no credit-card
utilisation on record" usually signals a thin file rather than an average one.

That reasoning is only partly borne out, and is reported as such: on its own the
missingness flag carries almost no marginal signal (0.99× lift), so its model
importance is interaction-driven. Collinear indicators were deduplicated — every
enquiry field is unreported for the same 6,321 borrowers, so their indicators
were identical columns.

---

## Model and evaluation

XGBoost, 400 trees, depth 5, learning rate 0.05, `scale_pos_weight` from the
training-split class ratio.

Evaluation choices worth stating:

- **Stratified 5-fold cross-validation, reported as mean ± sd.** A single 80/20
  split gives one number with no sense of its stability.
- **Average precision as the headline metric, not ROC-AUC.** At 16.8% positives,
  AUC is dominated by the majority class and reads optimistically.
- **Precision and recall at top-k**, because the operational task is working a
  ranked queue under fixed review capacity.
- **Feature statistics fitted on the training split only**, before any transform
  touches test rows.

### Ranking quality

Stress rate by score decile, held-out set, monotonic with no inversions:

| Decile | Accounts | Stressed | Stress rate | Lift |
|---|---|---|---|---|
| 1 (highest risk) | 1,027 | 487 | 47.4% | 2.83× |
| 2 | 1,027 | 343 | 33.4% | 1.99× |
| 3 | 1,027 | 239 | 23.3% | 1.39× |
| 10 (lowest risk) | 1,027 | 7 | 0.7% | 0.04× |

Risk tiers escalate cleanly: Low 2.9%, Medium 13.0%, High 28.1%, Critical 50.8%.

### The exposure benchmark

| Model | Features | ROC-AUC | Avg precision | Precision @10% |
|---|---|---|---|---|
| Exposure only | 6 | 0.7031 | 0.3077 | 37.2% |
| Full model | 57 | 0.7744 | 0.4058 | 47.5% |

### Performance within comparable-exposure strata

| Trade lines | Borrowers | Stressed | Stress rate | ROC-AUC | Precision @10% |
|---|---|---|---|---|---|
| 1–2 | 5,211 | 554 | 10.6% | 0.7675 | 31.9% |
| 3–5 | 2,542 | 515 | 20.3% | 0.7416 | 49.2% |
| 6–10 | 1,372 | 326 | 23.8% | 0.7449 | 54.0% |
| 11+ | 1,143 | 326 | 28.5% | 0.7355 | 62.3% |

Mean within-stratum AUC is 0.7474 against 0.7744 overall. The 0.027 gap is the
portion of the overall figure attributable to ranking large credit files above
small ones rather than discriminating between comparable borrowers. A wide gap
would mean the exposure confound still dominates; this one is narrow, and
performance is consistent across strata.

---

## Direction of the arrow

This is the sharpest objection available against the project, and it is not fully
answerable with this data. Stating it precisely:

The label covers the last 12 months. So do many of the features. `enq_L3m`,
`Total_TL_opened_L6M`, `pct_tl_open_L6M`, `closure_rate` and nine others all
measure activity inside the same window the outcome is measured in. A borrower who
went delinquent in month 3 may have made a burst of enquiries and opened accounts
in months 4–12 *because* they were already in distress. In that case the feature
is a **consequence** of the outcome, not a signal preceding it, and the model is
detecting distress rather than anticipating it.

The source is a single cross-sectional snapshot with no as-of dates. The ordering
of events within the window cannot be recovered, so **the direction is formally
unidentifiable.** No amount of modelling fixes that; only different data does.

What can be done is bound the exposure to it. Retraining with all 14 same-window
features removed gives a model that provably cannot be reading post-outcome
behaviour:

| Model | Features | ROC-AUC | Avg precision | Precision @10% |
|---|---|---|---|---|
| Exposure only (floor) | 6 | 0.7031 | 0.3077 | 37.2% |
| **No same-window features** | **43** | **0.7340** | 0.3436 | 41.7% |
| Full model | 57 | 0.7744 | 0.4058 | 47.5% |

Read honestly: of the **+0.0713** the full model gains over the exposure floor,
**57% (0.0404) depends on features sharing the label's window** and could in
principle be post-outcome behaviour. The remaining **43% (+0.0309)** comes from
lifetime stocks, point-in-time utilisation and borrower profile, which cannot be.

So the defensible claim is narrower than the headline: a model built only from
information that cannot be a reaction to the outcome still ranks deterioration at
AUC 0.734, meaningfully above an account-counting floor of 0.703. The full model's
0.774 should be read as **concurrent detection**, not anticipation.

Settling it properly requires panel data — features observed as of month 0,
outcomes recorded over months 1–12. That is the single most valuable next step for
this project and it is a data-acquisition problem, not a modelling one.

---

## A finding that did not survive

Under the lifetime label, stress rate rose monotonically with the number of gold
loans held — 7.0% at none to 53.2% at 40 or more, a 5.0× gradient. It looked like
the strongest result in the project, with a clean domain story: repeated pledging
and redeeming of gold is a recognised working-capital bridge for small Indian
traders. Gold loans are a collateral product, not a delinquency field, so label
leakage could not explain it.

**Exposure could.** Gold-loan count correlates strongly with total trade lines,
and the lifetime label rose with trade-line count. Under the fixed-window label
the marginal gradient falls to 1.69×, and holding trade-line count fixed it
flattens and then reverses. Within the largest exposure stratum:

| Gold loans held (11+ trade lines) | Borrowers | Stress rate |
|---|---|---|
| None | 1,435 | 26.8% |
| 1–4 | 904 | 30.6% |
| 5–9 | 1,059 | 29.6% |
| 10+ | 2,189 | **22.8%** |

Borrowers with 10+ gold loans are *less* likely to deteriorate than same-exposure
peers with none. This is Simpson's paradox — two variables moving together with a
third that drives the outcome. A plausible reading is that gold loans indicate
pledgeable collateral rather than distress, but that is a hypothesis, not a
result, and it is not claimed here.

The finding is documented rather than deleted. It was the most attractive claim in
the project, and the only reason it was caught is that the exposure control was
built before the claim was published.

---

## What the model relies on

Top features by gain: `Tot_Active_TL` (0.092), `closure_rate` (0.047),
`pct_tl_open_L6M` (0.042), `Total_TL` (0.040), `GL_Flag` (0.039),
`CC_enq_missing` (0.039), `open_close_imbalance` (0.024), `tot_enq` (0.023).

Gain splits credit among correlated features, so families are more informative
than individual ranks. The signal concentrates in **portfolio dynamics** — how
many active lines, how fast they open and close, the balance between the two —
with credit-seeking behaviour and data-availability flags secondary.

This contradicts an earlier version of this report, which claimed
`enquiry_acceleration` was the top predictor and a causal driver of stress, with a
28.7 percentage-point causal effect estimated via DoWhy. That estimate was
computed against the discarded approval-tier label and does not carry over; under
the current label and leakage boundary, `enquiry_acceleration` ranks 13th by gain.
The causal analysis is **withdrawn** rather than re-reported: re-running it would
require re-specifying the causal graph against the new label. That is future work,
not a current finding.

---

## Withdrawn from earlier versions

Kept as an explicit list so the git history and the notebooks can be read against
this report.

| Claim | Status |
|---|---|
| "Predicting stress 90 days early" | **Withdrawn.** Cross-sectional snapshot, no default timestamps. Nothing supports a 90-day horizon. |
| AUC 0.825 | **Superseded.** Computed on the approval-tier label. |
| AUC 0.8135 | **Superseded.** Computed on the lifetime label; only +0.025 above an account-counting baseline. |
| Enquiry acceleration as top predictor and causal driver (+28.7pp) | **Withdrawn.** Artefact of the discarded label; now 13th by gain. |
| Gold-loan gradient, 5.0× | **Withdrawn.** Confounded by exposure; reverses within strata. |
| Cox proportional hazards, C-index 0.671 | **Withdrawn.** Durations were simulated from the same variables used as predictors, making the concordance circular. |
| LSTM autoencoder anomaly detection, AUC 0.692 | **Not carried forward.** Computed on the discarded label. The architecture also used a single timestep, making it a dense autoencoder rather than a temporal model. |
| Conformal prediction, 90.3% coverage | **Not carried forward.** Computed on the discarded label, and the calibration was never persisted, so it could not be served. Worth redoing — the "route uncertain cases to manual review" output is the most operationally useful thing the earlier work produced. |
| Commodity volatility as a model input | **Dropped.** Collapsed to a single constant across all rows. |

---

## Domain grounding

Feature selection and the interpretation of portfolio-churn signals were checked
against wholesale trading operations in India's spice and herbs sector — actual
MSME cash-flow mechanics, payment cycles and working-capital bridging. This
informed which behaviours were treated as plausible distress signals; it is
qualitative context for the feature design, not evidence for any result reported
above.

---

## Limitations

1. **Concurrent detection, not a forward-dated forecast.** The data is a
   cross-sectional snapshot with no as-of dates. No lead-time claim is made, and
   because feature and label windows overlap, the direction of association is
   unidentifiable for 57% of the model's margin over the exposure floor. See
   "Direction of the arrow" — this is the project's binding limitation.

2. **Exposure is reduced, not eliminated.** The fixed window cut the prevalence
   gradient to 2.39×, but borrowers with more trade lines still show higher
   stress rates. Some of that is likely real risk. Within-stratum metrics are
   reported so the reader can judge.

3. **Retail proxy for firm risk.** Bureau records for individuals stand in for
   MSME proprietor risk. Not validated against firm-level financials, GST filing
   behaviour, or TReDS payment data.

4. **Utilisation fields are mostly absent.** 92.8% and 86.6% sentinel rates mean
   utilisation should be read as a weak input.

5. **Event count.** 16.76% prevalence gives roughly 1,720 positives in the test
   set; top-decile metrics rest on 487 of them. Fold-level variation is reported
   rather than a single split.

6. **A source data inconsistency.** `HL_Flag` is set for some borrowers who have
   `Home_TL = 0`. The contradiction is in the source bank extract and is not
   corrected here.

---

## Reproducing

```
python -m src.train      # label, features, model, metrics, scored portfolio
streamlit run app.py     # dashboard over the persisted artifacts
```

`src/config.py` holds the label definition and the leakage boundary.
`src/labels.py` builds the label and the exposure strata. `src/features.py`
implements fit/transform with persisted parameters. `src/train.py` runs
cross-validation, the exposure baseline, the stratified evaluation and the cohort
diagnostics.

---

## Conclusion

A model restricted to non-repayment signals ranks 12-month credit deterioration
with AUC 0.7744 and 47.5% precision in the top decile against a 16.8% base rate —
2.83× lift — consistently within exposure strata, beating an account-counting
baseline by 0.0713 AUC.

Restricted to features that cannot be a reaction to the outcome, it still ranks at
AUC 0.7340 against that same 0.7031 floor — which is the narrower claim the data
actually supports.

The more useful outcome is methodological. Two earlier versions of this project
reported higher numbers on weaker foundations: one on a label that did not measure
stress, one on a label that partly measured credit-file size. Both looked better
by AUC. The diagnostics that exposed them — comparing against an exposure-only
baseline, checking whether findings survive within strata, and retraining without
same-window features — are cheap to run, and were the difference between a
plausible result and a defensible one.

The remaining limitation is not fixable by modelling. Feature and label windows
overlap, so more than half the model's edge over an account-counting baseline
cannot be established as predictive rather than reactive. Resolving that needs
panel data with as-of dates, and until it exists the honest description of this
system is a triage tool that detects concurrent deterioration — not an early
warning that anticipates it.
