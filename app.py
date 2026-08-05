"""MSME Credit Stress Early-Warning Dashboard.

Reads only persisted artifacts (model, feature params, scored portfolio, metrics)
and calls the same src.features.transform used in training, so a borrower scores
identically here and in the training pipeline.

Run from the project root:  streamlit run app.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from src import config as C
from src import features, labels

st.set_page_config(
    page_title="MSME Stress Early-Warning",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Muted palette that holds up in both Streamlit themes.
TIER_COLORS = {
    "Low": "#2f9e6b",
    "Medium": "#c9a227",
    "High": "#d1730f",
    "Critical": "#c0392b",
}
ACCENT = "#4c78a8"

st.markdown(
    """
<style>
.block-container { padding-top: 2.2rem; max-width: 1400px; }
[data-testid="stMetricValue"] { font-size: 1.55rem; }
.caption-box {
    border-left: 3px solid #4c78a8;
    padding: 0.6rem 0.9rem;
    margin: 0.5rem 0 1.2rem 0;
    font-size: 0.88rem;
    opacity: 0.85;
}
.limitation {
    border-left: 3px solid #c9a227;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── artifact loading ────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    bundle = joblib.load(C.MODEL_PATH)
    return bundle["model"], bundle["feature_names"]


@st.cache_resource(show_spinner=False)
def load_feature_params():
    return features.load_params(C.FEATURE_PARAMS)


@st.cache_resource(show_spinner=False)
def load_explainer():
    model, _ = load_model()
    return shap.TreeExplainer(model)


@st.cache_data(show_spinner=False)
def load_metrics():
    with open(C.METRICS_PATH) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_scored():
    df = pd.read_csv(C.SCORED_PORTFOLIO)
    df["risk_tier"] = pd.Categorical(
        df["risk_tier"], categories=["Low", "Medium", "High", "Critical"], ordered=True
    )
    return df


@st.cache_data(show_spinner=False)
def load_raw_for_ids(ids: tuple[int, ...]) -> pd.DataFrame:
    """Raw rows for the scored borrowers only, so the 14 MB source file is read
    once and the app holds a small slice rather than the whole table."""
    raw = pd.read_csv(C.MERGED_DATA)
    return raw[raw[C.ID_COL].isin(ids)].set_index(C.ID_COL, drop=False)


@st.cache_data(show_spinner=False)
def build_feature_frame(ids: tuple[int, ...]) -> pd.DataFrame:
    raw = load_raw_for_ids(ids)
    params = load_feature_params()
    _, names = load_model()
    X = features.transform(raw, params)
    X.index = raw[C.ID_COL].to_numpy()
    return X[names]


try:
    MODEL, FEATURE_NAMES = load_model()
    METRICS = load_metrics()
    SCORED = load_scored()
except FileNotFoundError as exc:
    st.error(
        f"Missing artifact: `{exc.filename}`.\n\n"
        "Build it first from the project root:\n\n```\npython -m src.train\n```"
    )
    st.stop()

BASE_RATE = METRICS["test"]["base_rate"]
ID_TUPLE = tuple(SCORED[C.ID_COL].tolist())


# ── shared helpers ──────────────────────────────────────────────────

def queue_stats(df: pd.DataFrame, capacity_pct: float) -> dict:
    """What a credit team gets if it reviews the top `capacity_pct` by score."""
    n = max(1, int(len(df) * capacity_pct / 100))
    queue = df.nlargest(n, "stress_probability")
    caught = int(queue["actual_stress"].sum())
    total = int(df["actual_stress"].sum())
    return {
        "reviewed": n,
        "caught": caught,
        "missed": total - caught,
        "precision": caught / n,
        "recall": caught / total if total else 0.0,
        "lift": (caught / n) / BASE_RATE if BASE_RATE else 0.0,
        "cutoff": float(queue["stress_probability"].min()),
        "queue": queue,
    }


def tier_badge(tier: str) -> str:
    return f":{'green' if tier=='Low' else 'orange' if tier in ('Medium','High') else 'red'}[{tier}]"


# ── page: portfolio triage ──────────────────────────────────────────

def page_portfolio() -> None:
    st.subheader("Portfolio triage")
    st.markdown(
        '<div class="caption-box">Out-of-sample scores for '
        f"{len(SCORED):,} borrowers held out of training. Set your review "
        "capacity to see what a ranked queue actually returns at that "
        "workload.</div>",
        unsafe_allow_html=True,
    )

    capacity = st.slider(
        "Review capacity — share of the portfolio your team can manually review",
        min_value=1.0,
        max_value=25.0,
        value=10.0,
        step=0.5,
        format="%.1f%%",
    )
    q = queue_stats(SCORED, capacity)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accounts reviewed", f"{q['reviewed']:,}")
    c2.metric(
        "Precision", f"{q['precision']:.1%}", f"{q['lift']:.2f}x vs {BASE_RATE:.1%} base"
    )
    c3.metric("Stressed accounts caught", f"{q['caught']:,}", f"{q['recall']:.1%} of all")
    c4.metric("Missed", f"{q['missed']:,}", delta_color="inverse")
    c5.metric("Score cutoff", f"{q['cutoff']:.3f}")

    st.divider()
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("**Precision and recall across review capacities**")
        grid = np.arange(1, 26, 1.0)
        rows = [queue_stats(SCORED, c) for c in grid]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grid, y=[r["precision"] * 100 for r in rows],
                name="Precision", line=dict(color=ACCENT, width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grid, y=[r["recall"] * 100 for r in rows],
                name="Recall", line=dict(color="#e45756", width=2.5, dash="dot"),
            )
        )
        fig.add_hline(
            y=BASE_RATE * 100, line=dict(color="grey", dash="dash", width=1),
            annotation_text=f"base rate {BASE_RATE:.1%}", annotation_position="right",
        )
        fig.add_vline(x=capacity, line=dict(color="#888", width=1.5))
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Review capacity (% of portfolio)", yaxis_title="Percent",
            legend=dict(orientation="h", y=1.12, x=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Actual stress rate by risk tier**")
        tiers = (
            SCORED.groupby("risk_tier", observed=True)["actual_stress"]
            .agg(["count", "mean"])
            .reset_index()
        )
        fig2 = go.Figure(
            go.Bar(
                x=tiers["risk_tier"].astype(str),
                y=tiers["mean"] * 100,
                marker_color=[TIER_COLORS[t] for t in tiers["risk_tier"].astype(str)],
                text=[f"{v:.1%}<br>n={c:,}" for v, c in zip(tiers["mean"], tiers["count"])],
                textposition="outside",
            )
        )
        fig2.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="% actually stressed", showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, max(tiers["mean"]) * 130],
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown(f"**Review queue** — top {capacity:.1f}% by stress probability")
    view = q["queue"][[C.ID_COL, "stress_probability", "risk_tier", "actual_stress"]].copy()
    view["actual_stress"] = view["actual_stress"].map({1: "Stressed", 0: "Healthy"})
    st.dataframe(
        view.rename(
            columns={
                C.ID_COL: "Borrower ID",
                "stress_probability": "Stress probability",
                "risk_tier": "Tier",
                "actual_stress": "Actual outcome",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=320,
        column_config={
            "Stress probability": st.column_config.ProgressColumn(
                "Stress probability", min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )
    st.caption(
        "Actual outcome is shown because this is a held-out evaluation set. In "
        "production it would be unknown at scoring time."
    )


# ── page: borrower drill-down ───────────────────────────────────────

def page_borrower() -> None:
    st.subheader("Borrower drill-down")
    st.markdown(
        '<div class="caption-box">Select a borrower to see the score and the '
        "features driving it, then adjust inputs to test how the score responds."
        "</div>",
        unsafe_allow_html=True,
    )

    col_sel, col_score = st.columns([1, 2])

    with col_sel:
        borrower_id = st.selectbox(
            "Borrower ID",
            options=SCORED[C.ID_COL].tolist(),
            index=0,
            help="Sorted by stress probability, highest first.",
        )
        st.caption(f"{len(SCORED):,} scored borrowers available")

    X = build_feature_frame(ID_TUPLE)
    row = SCORED[SCORED[C.ID_COL] == borrower_id].iloc[0]
    x = X.loc[[borrower_id]]

    with col_score:
        m1, m2, m3 = st.columns(3)
        m1.metric("Stress probability", f"{row['stress_probability']:.3f}")
        m2.metric("Risk tier", str(row["risk_tier"]))
        m3.metric(
            "Actual outcome",
            "Stressed" if row["actual_stress"] == 1 else "Healthy",
        )
        pct = (SCORED["stress_probability"] < row["stress_probability"]).mean()
        st.caption(
            f"Riskier than {pct:.1%} of the portfolio. "
            f"Tier boundaries: Low <0.25, Medium <0.50, High <0.75, Critical >=0.75."
        )

    st.divider()

    tab_explain, tab_whatif = st.tabs(["Why this score", "What-if analysis"])

    with tab_explain:
        explainer = load_explainer()
        sv = explainer(x)
        contrib = (
            pd.DataFrame(
                {"feature": FEATURE_NAMES, "shap": sv.values[0], "value": x.iloc[0].to_numpy()}
            )
            .assign(abs_shap=lambda d: d["shap"].abs())
            .nlargest(14, "abs_shap")
            .sort_values("shap")
        )
        fig = go.Figure(
            go.Bar(
                x=contrib["shap"],
                y=[f"{f}  =  {v:,.3g}" for f, v in zip(contrib["feature"], contrib["value"])],
                orientation="h",
                marker_color=["#c0392b" if s > 0 else "#2f9e6b" for s in contrib["shap"]],
            )
        )
        fig.update_layout(
            height=460, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="SHAP contribution to log-odds  (red = raises risk)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "SHAP values are computed live for the selected borrower, so this "
            "explanation is specific to this account rather than a portfolio average."
        )

    with tab_whatif:
        st.markdown(
            "Adjust the model's strongest drivers and rescore. Useful for asking "
            "what would have to change for an account to leave the review queue."
        )
        importance = METRICS["feature_importance"]
        top_drivers = [f for f in sorted(importance, key=importance.get, reverse=True)][:6]

        modified = x.copy()
        cols = st.columns(3)
        for i, feat in enumerate(top_drivers):
            series = X[feat]
            lo, hi = float(series.quantile(0.01)), float(series.quantile(0.99))
            if hi <= lo:
                hi = lo + 1.0
            current = float(x.iloc[0][feat])
            with cols[i % 3]:
                modified[feat] = st.slider(
                    feat,
                    min_value=lo,
                    max_value=hi,
                    value=float(np.clip(current, lo, hi)),
                    key=f"whatif_{feat}",
                )

        new_prob = float(MODEL.predict_proba(modified)[:, 1][0])
        delta = new_prob - row["stress_probability"]
        w1, w2 = st.columns(2)
        w1.metric("Original probability", f"{row['stress_probability']:.3f}")
        w2.metric("Adjusted probability", f"{new_prob:.3f}", f"{delta:+.3f}")
        if abs(delta) < 1e-9:
            st.caption("Sliders are at the borrower's actual values.")


# ── page: batch scoring ─────────────────────────────────────────────

def page_batch() -> None:
    st.subheader("Batch scoring")
    st.markdown(
        '<div class="caption-box">Upload raw borrower records in the same '
        "column layout as <code>credit_merged_clean.csv</code>. Feature "
        "engineering, imputation and scoring use the persisted training "
        "parameters, so results match the training pipeline exactly.</div>",
        unsafe_allow_html=True,
    )

    upload = st.file_uploader("CSV of raw borrower records", type=["csv"])
    if upload is None:
        st.info(
            "No file uploaded. The scored held-out set is available on the "
            "Portfolio triage page."
        )
        return

    raw = pd.read_csv(upload)
    st.write(f"Loaded **{len(raw):,}** rows, {raw.shape[1]} columns.")

    params = load_feature_params()
    try:
        X = features.transform(raw, params)
        missing = [c for c in FEATURE_NAMES if c not in X.columns]
        if missing:
            st.error(
                f"{len(missing)} required feature(s) could not be built from this "
                f"file — first few: {missing[:5]}"
            )
            return
        probs = MODEL.predict_proba(X[FEATURE_NAMES])[:, 1]
    except KeyError as exc:
        st.error(f"Required source column missing from the upload: {exc}")
        return

    out = pd.DataFrame(
        {
            C.ID_COL: raw[C.ID_COL] if C.ID_COL in raw.columns else np.arange(len(raw)),
            "stress_probability": probs,
        }
    )
    out["risk_tier"] = pd.cut(
        out["stress_probability"],
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Low", "Medium", "High", "Critical"],
    )
    out = out.sort_values("stress_probability", ascending=False)

    # If the upload carries the source delinquency columns, the true label can be
    # derived for validation. Never used as an input to the score.
    if C.DPD_60_COL in raw.columns and all(c in raw.columns for c in C.NPA_COUNT_COLS):
        actual = labels.build_stress_label(raw)
        out["actual_stress"] = actual.to_numpy()[out.index]
        st.caption(
            "This file contains the delinquency columns, so the true label was "
            "derived for validation. It is excluded from scoring by the leakage rule."
        )

    k1, k2, k3 = st.columns(3)
    k1.metric("Scored", f"{len(out):,}")
    k2.metric(
        "High or Critical",
        f"{int(out['risk_tier'].isin(['High','Critical']).sum()):,}",
    )
    k3.metric("Mean probability", f"{out['stress_probability'].mean():.3f}")

    st.dataframe(out, use_container_width=True, hide_index=True, height=380)
    st.download_button(
        "Download scored file",
        out.to_csv(index=False).encode(),
        file_name="scored_borrowers.csv",
        mime="text/csv",
    )


# ── page: model card ────────────────────────────────────────────────

def page_model_card() -> None:
    st.subheader("Model card")

    cv = METRICS["cv"]["summary"]
    test = METRICS["test"]

    st.markdown("**Label definition**")
    st.code(METRICS["label_definition"], language="text")
    st.markdown(
        "Under RBI's asset-classification norms an account is tracked as SMA-0 "
        "(1-30 days overdue), SMA-1 (31-60), SMA-2 (61-90) and NPA (90+). SMA-2 "
        "is the supervisory threshold at which Indian lenders must report and act "
        "on a stressed account, which makes it the operational definition of "
        "stress used here."
    )

    st.divider()
    st.markdown("**Performance** — cross-validated on the training split, then "
                "confirmed once on a held-out test set")

    perf = pd.DataFrame(
        [
            {
                "Metric": "ROC-AUC",
                "Cross-validation": f"{cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}",
                "Held-out test": f"{test['auc']:.4f}",
            },
            {
                "Metric": "Average precision",
                "Cross-validation": f"{cv['ap_mean']:.4f} ± {cv['ap_std']:.4f}",
                "Held-out test": f"{test['ap']:.4f}",
            },
            {
                "Metric": "Precision @ top 5%",
                "Cross-validation": f"{cv['precision_top5_mean']:.1%} ± {cv['precision_top5_std']:.1%}",
                "Held-out test": f"{test['precision_top5']:.1%}",
            },
            {
                "Metric": "Precision @ top 10%",
                "Cross-validation": f"{cv['precision_top10_mean']:.1%} ± {cv['precision_top10_std']:.1%}",
                "Held-out test": f"{test['precision_top10']:.1%}",
            },
            {
                "Metric": "Recall @ top 10%",
                "Cross-validation": f"{cv['recall_top10_mean']:.1%} ± {cv['recall_top10_std']:.1%}",
                "Held-out test": f"{test['recall_top10']:.1%}",
            },
        ]
    )
    st.dataframe(perf, use_container_width=True, hide_index=True)
    st.caption(
        f"Base stress rate is {BASE_RATE:.2%}, so average precision should be "
        f"read against a {BASE_RATE:.4f} baseline, not against 0.5. Average "
        "precision is the headline metric here because ROC-AUC flatters models "
        "on imbalanced data."
    )

    st.divider()
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown("**Lift by score decile** (held-out test set)")
        dec = pd.DataFrame(METRICS["deciles"])
        fig = go.Figure(
            go.Bar(
                x=dec["decile"].astype(str), y=dec["lift"],
                marker_color=ACCENT,
                text=[f"{v:.2f}x" for v in dec["lift"]], textposition="outside",
            )
        )
        fig.add_hline(y=1.0, line=dict(color="grey", dash="dash"))
        fig.update_layout(
            height=330, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Decile (1 = highest risk)", yaxis_title="Lift vs base rate",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Top 15 features by gain**")
        imp = METRICS["feature_importance"]
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:15][::-1]
        fig2 = go.Figure(
            go.Bar(
                x=[v for _, v in top], y=[k for k, _ in top],
                orientation="h", marker_color="#6a8cba",
            )
        )
        fig2.update_layout(
            height=330, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Gain", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("**Leakage controls**")
    st.markdown(
        f"The label is built from delinquency and asset-classification fields, so "
        f"all {len(METRICS['excluded_for_leakage'])} columns that define or "
        "mechanically imply it are excluded from the feature set by rule, and the "
        "training script asserts the exclusion rather than trusting it. Feature "
        "statistics — medians, clip bounds, normalisers — are fitted on the "
        "training split only and persisted to JSON, so the dashboard and the "
        "training pipeline produce identical features for the same borrower."
    )
    with st.expander(f"View all {len(METRICS['excluded_for_leakage'])} excluded columns"):
        st.write(", ".join(f"`{c}`" for c in METRICS["excluded_for_leakage"]))

    st.divider()
    st.markdown("**Limitations**")
    for text in [
        "The model predicts <b>concurrent</b> stress classification, not a "
        "forward-dated event. The source data is a cross-sectional snapshot with "
        "no default timestamps, so no lead-time claim is made or supported.",
        "<code>CC_utilization</code> is 92.8% unreported in the source and "
        "<code>PL_utilization</code> 86.6%. Both are median-filled and paired with "
        "an explicit missingness indicator; the indicators carry more signal than "
        "the filled values, so utilisation should be read as a weak input here.",
        "Trained on retail credit-bureau records used as a proxy for MSME "
        "proprietor risk. It is not validated on firm-level financials, GST filing "
        "behaviour, or TReDS payment data.",
        "Commodity price volatility is computed cross-sectionally rather than as a "
        "time series, because the Agmarknet extract covers a limited window. It is "
        "not used as a model input.",
        "A 10.6% event rate means the top-decile metrics rest on roughly 370 "
        "stressed accounts in the test set. Fold-level variation is reported above "
        "rather than a single split.",
    ]:
        st.markdown(f'<div class="limitation">{text}</div>', unsafe_allow_html=True)


# ── shell ───────────────────────────────────────────────────────────

PAGES = {
    "Portfolio triage": page_portfolio,
    "Borrower drill-down": page_borrower,
    "Batch scoring": page_batch,
    "Model card": page_model_card,
}

with st.sidebar:
    st.markdown("### MSME Stress Early-Warning")
    st.caption("RBI SMA-2 / NPA classification risk")
    choice = st.radio("View", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.metric("Scored borrowers", f"{len(SCORED):,}")
    st.metric("Base stress rate", f"{BASE_RATE:.2%}")
    st.metric("Test ROC-AUC", f"{METRICS['test']['auc']:.3f}")
    st.metric("Features", METRICS["n_features"])
    st.divider()
    st.caption(
        "Scores shown are out-of-sample, from borrowers held out of model training."
    )

st.title("MSME Credit Stress Early-Warning")
PAGES[choice]()
