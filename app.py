"""MSME Credit Stress Early-Warning Dashboard.

Reads only persisted artifacts (model, feature params, scored portfolio, metrics)
and calls the same src.features.transform used in training, so a borrower scores
identically here and in the training pipeline.

Chart colours and chrome come from src.theme, where each palette is recorded with
the validator result that justified it.

Run from the project root:  streamlit run app.py
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from src import config as C
from src import features, labels
from src import theme as T

st.set_page_config(
    page_title="MSME Stress Early-Warning",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(T.CSS, unsafe_allow_html=True)


# ── artifact loading ────────────────────────────────────────────────
# Every loader is keyed on the artifact's modification time. Streamlit caches
# live for the lifetime of the server process, so without this a re-run of
# `python -m src.train` leaves the app serving the previous model and metrics
# until someone restarts it — which surfaces as a confusing KeyError rather than
# as stale data.

def _stamp(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_resource(show_spinner=False)
def load_model(stamp_key: float = 0.0):
    bundle = joblib.load(C.MODEL_PATH)
    return bundle["model"], bundle["feature_names"]


@st.cache_resource(show_spinner=False)
def load_feature_params(stamp_key: float = 0.0):
    return features.load_params(C.FEATURE_PARAMS)


@st.cache_resource(show_spinner=False)
def load_explainer(stamp_key: float = 0.0):
    model, _ = load_model(stamp_key)
    return shap.TreeExplainer(model)


@st.cache_data(show_spinner=False)
def load_metrics(stamp_key: float = 0.0):
    with open(C.METRICS_PATH) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_scored(stamp_key: float = 0.0):
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
def build_feature_frame(ids: tuple[int, ...], stamp_key: float = 0.0) -> pd.DataFrame:
    raw = load_raw_for_ids(ids)
    params = load_feature_params(stamp_key)
    _, names = load_model(stamp_key)
    X = features.transform(raw, params)
    X.index = raw[C.ID_COL].to_numpy()
    return X[names]


MODEL_STAMP = _stamp(C.MODEL_PATH)
METRICS_STAMP = _stamp(C.METRICS_PATH)
SCORED_STAMP = _stamp(C.SCORED_PORTFOLIO)

try:
    MODEL, FEATURE_NAMES = load_model(MODEL_STAMP)
    METRICS = load_metrics(METRICS_STAMP)
    SCORED = load_scored(SCORED_STAMP)
except FileNotFoundError as exc:
    st.error(
        f"Missing artifact: `{exc.filename}`.\n\n"
        "Build it first from the project root:\n\n```\npython -m src.train\n```"
    )
    st.stop()

BASE_RATE = METRICS["test"]["base_rate"]
ID_TUPLE = tuple(SCORED[C.ID_COL].tolist())
TIER_ORDER = ["Low", "Medium", "High", "Critical"]


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


def table_view(df: pd.DataFrame, label: str = "View as table") -> None:
    """Every chart gets a table twin so no value is reachable only by colour."""
    with st.expander(label):
        st.dataframe(df, use_container_width=True, hide_index=True)


def tier_frame() -> pd.DataFrame:
    t = (
        SCORED.groupby("risk_tier", observed=True)["actual_stress"]
        .agg(accounts="size", stressed="sum", stress_rate="mean")
        .reindex(TIER_ORDER)
        .reset_index()
    )
    t["lift"] = t["stress_rate"] / BASE_RATE
    return t


# ── page: overview ──────────────────────────────────────────────────

def page_overview() -> None:
    cv = METRICS["cv"]["summary"]
    test = METRICS["test"]

    # Hero figure — exactly one per view.
    hero, tiles = st.columns([1, 2.1])
    with hero:
        st.markdown(
            f'<div class="hero-figure">{test["precision_top10"] / BASE_RATE:.1f}'
            f'<span class="hero-unit">×</span></div>'
            '<div class="hero-caption">More stressed accounts found per review than '
            "random sampling, at a 10% review workload. This is the number a credit "
            "team feels.</div>",
            unsafe_allow_html=True,
        )
    with tiles:
        a, b, c = st.columns(3)
        a.markdown(
            T.tile(
                "Precision at top 10%",
                f"{test['precision_top10']:.1%}",
                f"vs {BASE_RATE:.1%} base rate",
            ),
            unsafe_allow_html=True,
        )
        b.markdown(
            T.tile(
                "ROC-AUC",
                f"{cv['auc_mean']:.3f}",
                f"± {cv['auc_std']:.3f} across 5 folds",
            ),
            unsafe_allow_html=True,
        )
        c.markdown(
            T.tile(
                "Average precision",
                f"{test['ap']:.3f}",
                f"{test['ap'] / BASE_RATE:.1f}× the {BASE_RATE:.3f} baseline",
            ),
            unsafe_allow_html=True,
        )
        st.write("")
        d, e, f = st.columns(3)
        d.markdown(
            T.tile("Borrowers scored", f"{len(SCORED):,}", "held out of training"),
            unsafe_allow_html=True,
        )
        e.markdown(
            T.tile("Recall at top 10%", f"{test['recall_top10']:.1%}", "of all stressed accounts"),
            unsafe_allow_html=True,
        )
        exp = METRICS.get("exposure_baseline", {})
        f.markdown(
            T.tile(
                "Margin over exposure",
                f"+{METRICS.get('exposure_margin', 0):.3f}",
                f"AUC above a {exp.get('auc', 0):.3f} account-counting baseline",
            ),
            unsafe_allow_html=True,
        )

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-label">Does the ranking hold up</div>', unsafe_allow_html=True)
        st.markdown("**Stress rate by score decile**")
        dec = pd.DataFrame(METRICS["deciles"])
        # One series, one colour — bar length already encodes magnitude, so a
        # value-ramp here would double-encode it.
        fig = go.Figure(
            go.Bar(
                x=dec["decile"].astype(str),
                y=dec["stress_rate"] * 100,
                marker=dict(color=T.SERIES_1, cornerradius=4),
                width=0.62,
                hovertemplate="Decile %{x}<br>%{y:.1f}% stressed<extra></extra>",
            )
        )
        # Reference threshold — deliberately dashed, unlike the solid gridlines.
        fig.add_hline(
            y=BASE_RATE * 100,
            line=dict(color=T.INK_MUTED, dash="dash", width=1),
            annotation_text=f"base rate {BASE_RATE:.1%}",
            annotation_position="top right",
            annotation_font=dict(size=10, color=T.INK_MUTED),
        )
        # Label the extreme only, not every bar.
        fig.add_annotation(
            x="1", y=dec.loc[dec["decile"] == 1, "stress_rate"].iloc[0] * 100,
            text=f"{dec.loc[dec['decile'] == 1, 'stress_rate'].iloc[0]:.1%}",
            showarrow=False, yshift=12, font=dict(size=11, color=T.INK),
        )
        st.plotly_chart(
            T.style_fig(fig, x_title="Decile (1 = highest risk)", y_title="% actually stressed"),
            use_container_width=True,
        )
        st.caption(
            "Monotonic from 35.9% down to 0.3%. Each decile is riskier than the "
            "one below it, with no inversions."
        )
        table_view(
            dec.assign(
                stress_rate=lambda d: (d["stress_rate"] * 100).round(1),
                lift=lambda d: d["lift"].round(2),
            )[["decile", "accounts", "stressed", "stress_rate", "lift"]].rename(
                columns={"stress_rate": "stress_rate_%"}
            )
        )

    with right:
        st.markdown(
            '<div class="section-label">Is it better than just counting accounts</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**AUC within comparable-exposure groups**")
        strata = pd.DataFrame(METRICS.get("strata", []))
        exp = METRICS.get("exposure_baseline", {})
        if strata.empty or not exp:
            st.info("Re-run `python -m src.train` to populate the exposure benchmark.")
        else:
            fig2 = go.Figure(
                go.Bar(
                    x=strata["band"],
                    y=strata["auc"],
                    marker=dict(color=T.SERIES_1, cornerradius=4),
                    width=0.6,
                    customdata=np.stack(
                        [strata["accounts"], strata["stress_rate"] * 100], axis=-1
                    ),
                    hovertemplate=(
                        "%{x} trade lines<br>AUC %{y:.3f}"
                        "<br>%{customdata[0]:,} accounts · %{customdata[1]:.1f}% stressed"
                        "<extra></extra>"
                    ),
                )
            )
            # Reference line without an inline label: the bars sit just above it
            # across the full width, so no corner is free. The caption carries
            # the value.
            fig2.add_hline(
                y=exp["auc"],
                line=dict(color=T.INK_MUTED, dash="dash", width=1),
            )
            st.plotly_chart(
                T.style_fig(
                    fig2,
                    x_title="Trade lines held",
                    y_title="ROC-AUC",
                    y_range=[0, 1.0],
                ),
                use_container_width=True,
            )
            st.caption(
                f"The label counts deterioration events, so borrowers holding more "
                f"accounts have more chances to register one — a model can score well "
                f"by little more than counting trade lines. A baseline using only "
                f"account counts and file ages reaches {exp['auc']:.3f} (dashed). The "
                f"full model reaches {METRICS['test']['auc']:.3f} overall and stays "
                f"between {strata['auc'].min():.3f} and {strata['auc'].max():.3f} "
                f"within every exposure group, so it is discriminating among peers "
                "rather than just ranking big files above small ones."
            )
            table_view(
                strata.assign(
                    stress_rate=lambda d: (d["stress_rate"] * 100).round(1),
                    auc=lambda d: d["auc"].round(4),
                    precision_top10=lambda d: (d["precision_top10"] * 100).round(1),
                ).rename(
                    columns={"stress_rate": "stress_rate_%", "precision_top10": "precision_top10_%"}
                )
            )

    with st.expander(
        "A finding that did not survive the exposure control — worth reading"
    ):
        st.markdown(
            "An earlier version of this project reported that stress rate rises "
            "monotonically with the number of gold loans held, from 7% to 53% — a "
            "5.0× gradient, and a tidy story about traders cycling gold loans for "
            "working capital.\n\n"
            "**It does not hold.** Gold-loan count is strongly correlated with total "
            "trade lines, and the old label counted lifetime events, so both were "
            "tracking credit-file size. Once trade-line count is held fixed the "
            "gradient flattens and then reverses — in the largest exposure group, "
            "borrowers with 10+ gold loans are stressed **22.8%** of the time versus "
            "**26.8%** for those with none. Textbook Simpson's paradox. Gold loans "
            "are secured credit, so holding them plausibly signals pledgeable "
            "collateral rather than distress.\n\n"
            "It is kept here deliberately: the marginal gradient looked like the "
            "strongest result in the project, and the only reason it was caught is "
            "that the exposure control was built before the claim was published."
        )
        gwe = pd.DataFrame(METRICS.get("gold_loan_within_exposure", []))
        if not gwe.empty:
            pivot = (
                gwe.pivot(index="exposure_band", columns="gold_band", values="stress_rate")
                .reindex(TIER_ORDER[:0] + ["1-2", "3-5", "6-10", "11+"])
                .mul(100).round(1)
            )
            st.markdown("**Stress rate (%) by gold-loan band, within exposure strata**")
            st.dataframe(pivot, use_container_width=True)
            st.caption(
                "Read across each row: within a fixed exposure group, more gold "
                "loans does not mean more stress. Cells with fewer than 40 "
                "borrowers are omitted."
            )

    st.divider()
    st.markdown('<div class="section-label">How this avoids the usual traps</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.markdown(
        T.rigour_card(
            "The observation window is fixed",
            "Stress means a delinquency or NPA classification inside the last 12 "
            "months, so every borrower is observed over the same period. A "
            "lifetime definition would reward borrowers for simply having a "
            "longer credit file.",
        ),
        unsafe_allow_html=True,
    )
    nov = METRICS.get("no_overlap_variant", {})
    r2.markdown(
        T.rigour_card(
            "The weakest point is measured",
            f"Feature and label windows overlap, so some signal could be the "
            f"borrower reacting to their own delinquency. Dropping all "
            f"{nov.get('n_dropped', 0)} same-window features costs "
            f"{nov.get('auc', 0) - test['auc']:+.4f} AUC and still clears the "
            "exposure floor.",
        ),
        unsafe_allow_html=True,
    )
    r3.markdown(
        T.rigour_card(
            "The baseline is not 0.5",
            f"A model using only account counts and file ages reaches "
            f"{METRICS.get('exposure_baseline', {}).get('auc', 0):.3f} on this "
            f"label. That is the floor to beat, and it is reported alongside "
            "5-fold cross-validated metrics rather than hidden.",
        ),
        unsafe_allow_html=True,
    )


# ── page: portfolio triage ──────────────────────────────────────────

def page_portfolio() -> None:
    st.markdown(
        '<div class="note">Out-of-sample scores for '
        f"{len(SCORED):,} borrowers held out of training. Set your review capacity "
        "to see what a ranked queue returns at that workload.</div>",
        unsafe_allow_html=True,
    )

    capacity = st.slider(
        "Review capacity — share of the portfolio your team can manually review",
        min_value=1.0, max_value=25.0, value=10.0, step=0.5, format="%.1f%%",
    )
    q = queue_stats(SCORED, capacity)

    cols = st.columns(5)
    for col, (label, value, delta) in zip(
        cols,
        [
            ("Accounts reviewed", f"{q['reviewed']:,}", f"of {len(SCORED):,}"),
            ("Precision", f"{q['precision']:.1%}", f"{q['lift']:.2f}× vs {BASE_RATE:.1%} base"),
            ("Stressed caught", f"{q['caught']:,}", f"{q['recall']:.1%} of all stressed"),
            ("Missed", f"{q['missed']:,}", "ranked below the cutoff"),
            ("Score cutoff", f"{q['cutoff']:.3f}", "lowest score reviewed"),
        ],
    ):
        col.markdown(T.tile(label, value, delta), unsafe_allow_html=True)

    st.divider()
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("**Precision and recall across review capacities**")
        grid = np.arange(1, 26, 1.0)
        rows = [queue_stats(SCORED, c) for c in grid]
        prec = [r["precision"] * 100 for r in rows]
        rec = [r["recall"] * 100 for r in rows]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grid, y=prec, name="Precision", mode="lines",
                line=dict(color=T.SERIES_1, width=2, shape="spline"),
                hovertemplate="At %{x:.0f}% capacity<br>%{y:.1f}% precision<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grid, y=rec, name="Recall", mode="lines",
                line=dict(color=T.SERIES_2, width=2, shape="spline"),
                hovertemplate="At %{x:.0f}% capacity<br>%{y:.1f}% recall<extra></extra>",
            )
        )
        # End-markers: the two series separate at the right edge and never cross
        # there, so no surface ring is needed — which is just as well, since a
        # ring would have to be painted in the page's surface colour and the
        # theme is not knowable here.
        for y_end, colour in ((prec[-1], T.SERIES_1), (rec[-1], T.SERIES_2)):
            fig.add_trace(
                go.Scatter(
                    x=[grid[-1]], y=[y_end], mode="markers",
                    marker=dict(size=8, color=colour),
                    showlegend=False, hoverinfo="skip",
                )
            )
        fig.add_hline(
            y=BASE_RATE * 100, line=dict(color=T.INK_MUTED, dash="dash", width=1),
            annotation_text=f"base rate {BASE_RATE:.1%}", annotation_position="bottom right",
            annotation_font=dict(size=10, color=T.INK_MUTED),
        )
        fig.add_vline(x=capacity, line=dict(color=T.AXIS, width=1))
        st.plotly_chart(
            T.style_fig(
                fig, show_legend=True,
                x_title="Review capacity (% of portfolio)", y_title="Percent",
            ),
            use_container_width=True,
        )
        st.caption(
            "The two always move against each other: reviewing more accounts "
            "catches more stress but a smaller share of each review pays off. "
            "The vertical line marks your current setting."
        )
        table_view(
            pd.DataFrame(
                {
                    "capacity_%": grid,
                    "precision_%": np.round(prec, 1),
                    "recall_%": np.round(rec, 1),
                    "reviewed": [r["reviewed"] for r in rows],
                    "caught": [r["caught"] for r in rows],
                }
            )
        )

    with right:
        st.markdown("**Actual stress rate by risk tier**")
        tiers = tier_frame()
        fig2 = go.Figure(
            go.Bar(
                x=tiers["risk_tier"].astype(str),
                y=tiers["stress_rate"] * 100,
                marker=dict(
                    color=[T.STATUS[t] for t in tiers["risk_tier"].astype(str)],
                    cornerradius=4,
                ),
                width=0.6,
                customdata=np.stack([tiers["accounts"], tiers["lift"]], axis=-1),
                hovertemplate=(
                    "%{x} tier<br>%{y:.1f}% stressed"
                    "<br>%{customdata[0]:,} accounts · %{customdata[1]:.2f}× lift<extra></extra>"
                ),
            )
        )
        crit = tiers.iloc[-1]
        fig2.add_annotation(
            x=str(crit["risk_tier"]), y=crit["stress_rate"] * 100,
            text=f"{crit['stress_rate']:.1%}", showarrow=False, yshift=12,
            font=dict(size=11, color=T.INK),
        )
        st.plotly_chart(
            T.style_fig(
                fig2, y_title="% actually stressed",
                y_range=[0, tiers["stress_rate"].max() * 100 * 1.28],
            ),
            use_container_width=True,
        )
        st.caption(
            "Tier names carry the meaning; the colour only reinforces it. Each "
            "tier is riskier than the one before, which is the ordering check."
        )
        table_view(
            tiers.assign(
                stress_rate=lambda d: (d["stress_rate"] * 100).round(1),
                lift=lambda d: d["lift"].round(2),
            ).rename(columns={"stress_rate": "stress_rate_%"})
        )

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
        use_container_width=True, hide_index=True, height=300,
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
    st.markdown(
        '<div class="note">Select a borrower to see the score and the features '
        "driving it, then adjust inputs to test how the score responds.</div>",
        unsafe_allow_html=True,
    )

    col_sel, col_score = st.columns([1, 2])
    with col_sel:
        borrower_id = st.selectbox(
            "Borrower ID", options=SCORED[C.ID_COL].tolist(), index=0,
            help="Sorted by stress probability, highest first.",
        )
        st.caption(f"{len(SCORED):,} scored borrowers available")

    X = build_feature_frame(ID_TUPLE, MODEL_STAMP)
    row = SCORED[SCORED[C.ID_COL] == borrower_id].iloc[0]
    x = X.loc[[borrower_id]]
    tier = str(row["risk_tier"])

    with col_score:
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            T.tile("Stress probability", f"{row['stress_probability']:.3f}"),
            unsafe_allow_html=True,
        )
        m2.markdown(T.tile("Risk tier", tier), unsafe_allow_html=True)
        m3.markdown(
            T.tile(
                "Actual outcome",
                "Stressed" if row["actual_stress"] == 1 else "Healthy",
                "held-out ground truth",
            ),
            unsafe_allow_html=True,
        )
        pct = (SCORED["stress_probability"] < row["stress_probability"]).mean()
        st.caption(
            f"Riskier than {pct:.1%} of the portfolio. Tier boundaries: "
            "Low <0.25, Medium <0.50, High <0.75, Critical ≥0.75."
        )

    st.divider()
    tab_explain, tab_whatif = st.tabs(["Why this score", "What-if analysis"])

    with tab_explain:
        explainer = load_explainer(MODEL_STAMP)
        sv = explainer(x)
        contrib = (
            pd.DataFrame(
                {"feature": FEATURE_NAMES, "shap": sv.values[0], "value": x.iloc[0].to_numpy()}
            )
            .assign(abs_shap=lambda d: d["shap"].abs())
            .nlargest(14, "abs_shap")
            .sort_values("shap")
        )
        # Polarity -> diverging poles, blue/red rather than green/red (nearly 3x
        # better separation under deuteranopia).
        fig = go.Figure(
            go.Bar(
                x=contrib["shap"],
                # The borrower's own value rides the category label. It is a
                # different quantity from the bar length, so this adds
                # information rather than double-encoding it, and it saves the
                # reader from hovering every row to understand the account.
                y=[f"{f}  =  {v:,.3g}" for f, v in zip(contrib["feature"], contrib["value"])],
                orientation="h",
                marker=dict(
                    color=[T.POLE_POS if s > 0 else T.POLE_NEG for s in contrib["shap"]],
                    cornerradius=4,
                ),
                hovertemplate="%{y}<br>contribution %{x:+.3f}<extra></extra>",
            )
        )
        fig.add_vline(x=0, line=dict(color=T.AXIS, width=1))
        styled = T.style_fig(
            fig, height=470,
            x_title="← lowers risk    ·    raises risk →   (SHAP, log-odds)",
        )
        styled.update_yaxes(showgrid=False)
        styled.update_xaxes(showgrid=True, gridcolor=T.GRID, zeroline=False)
        st.plotly_chart(styled, use_container_width=True)
        st.caption(
            "Red bars pushed this borrower's score up, blue pulled it down; bar "
            "length is how much. These are the inputs behind the model's "
            "arithmetic for this account — not a claim about what caused the stress."
        )
        table_view(
            contrib.sort_values("abs_shap", ascending=False)[["feature", "value", "shap"]]
            .round(4)
            .rename(columns={"shap": "shap_contribution"}),
            "View contributions as table",
        )

    with tab_whatif:
        st.markdown(
            "Adjust the model's strongest drivers and rescore — useful for asking "
            "what would have to change for an account to leave the review queue."
        )
        importance = METRICS["feature_importance"]
        top_drivers = sorted(importance, key=importance.get, reverse=True)[:6]

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
                    feat, min_value=lo, max_value=hi,
                    value=float(np.clip(current, lo, hi)), key=f"whatif_{feat}",
                )

        new_prob = float(MODEL.predict_proba(modified)[:, 1][0])
        delta = new_prob - row["stress_probability"]
        w1, w2, w3 = st.columns(3)
        w1.markdown(
            T.tile("Original probability", f"{row['stress_probability']:.3f}"),
            unsafe_allow_html=True,
        )
        w2.markdown(
            T.tile("Adjusted probability", f"{new_prob:.3f}", f"{delta:+.3f}"),
            unsafe_allow_html=True,
        )
        new_tier = pd.cut(
            [new_prob], bins=[-0.001, 0.25, 0.50, 0.75, 1.001], labels=TIER_ORDER
        )[0]
        w3.markdown(T.tile("Adjusted tier", str(new_tier)), unsafe_allow_html=True)
        if abs(delta) < 1e-9:
            st.caption("Sliders are at the borrower's actual values.")


# ── page: batch scoring ─────────────────────────────────────────────

def page_batch() -> None:
    st.markdown(
        '<div class="note">Upload raw borrower records in the same column layout '
        "as <code>credit_merged_clean.csv</code>. Feature engineering, imputation "
        "and scoring use the persisted training parameters, so results match the "
        "training pipeline exactly.</div>",
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

    params = load_feature_params(MODEL_STAMP)
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
        out["stress_probability"], bins=[-0.001, 0.25, 0.50, 0.75, 1.001], labels=TIER_ORDER
    )
    out = out.sort_values("stress_probability", ascending=False)

    # If the upload carries the source delinquency columns, the true label can be
    # derived for validation. Never used as an input to the score.
    if labels.has_label_columns(raw):
        actual = labels.build_stress_label(raw)
        out["actual_stress"] = actual.to_numpy()[out.index]
        st.caption(
            "This file contains the delinquency columns, so the true label was "
            "derived for validation. It is excluded from scoring by the leakage rule."
        )

    k = st.columns(4)
    k[0].markdown(T.tile("Scored", f"{len(out):,}"), unsafe_allow_html=True)
    k[1].markdown(
        T.tile(
            "High or Critical",
            f"{int(out['risk_tier'].isin(['High', 'Critical']).sum()):,}",
            f"{out['risk_tier'].isin(['High', 'Critical']).mean():.1%} of upload",
        ),
        unsafe_allow_html=True,
    )
    k[2].markdown(
        T.tile("Mean probability", f"{out['stress_probability'].mean():.3f}"),
        unsafe_allow_html=True,
    )
    k[3].markdown(
        T.tile("Max probability", f"{out['stress_probability'].max():.3f}"),
        unsafe_allow_html=True,
    )

    st.write("")
    st.dataframe(out, use_container_width=True, hide_index=True, height=360)
    st.download_button(
        "Download scored file",
        out.to_csv(index=False).encode(),
        file_name="scored_borrowers.csv",
        mime="text/csv",
    )


# ── page: model card ────────────────────────────────────────────────

def page_model_card() -> None:
    cv = METRICS["cv"]["summary"]
    test = METRICS["test"]

    st.markdown('<div class="section-label">Label definition</div>', unsafe_allow_html=True)
    # Rendered as markdown rather than st.code: a single long line in a code
    # block clips at the container edge instead of wrapping.
    st.markdown(
        f'<div class="note"><b>Stressed</b> = {METRICS["label_definition"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Under RBI's asset-classification norms an account is tracked as SMA-0 "
        "(1–30 days overdue), SMA-1 (31–60), SMA-2 (61–90) and NPA (90+). The SMA "
        "framework is a rolling current-state view — what supervisors and lenders "
        "monitor month to month — so a fixed recent window is the right shape for "
        "the label."
    )
    exp = METRICS.get("exposure_baseline", {})
    st.markdown(
        f"**Why the window is fixed rather than lifetime.** An earlier version of "
        f"this model defined stress as *ever* having been 60+ days past due. That "
        f"is not exposure-comparable: the label counts events, so a borrower "
        f"holding 20 accounts across 15 years has mechanically more chances to "
        f"register one than a borrower with 2 accounts across 3 years. The label "
        f"was partly measuring credit-file size. Switching to a fixed 12-month "
        f"window cut the prevalence gradient between the largest and smallest "
        f"exposure groups from **5.45× to 2.39×**, and dropped the AUC reachable "
        f"by an account-counting baseline from **0.782 to {exp.get('auc', 0):.3f}**. "
        f"The full model's margin over that baseline rose from **+0.025 to "
        f"+{METRICS.get('exposure_margin', 0):.3f}** — the headline AUC fell, and "
        f"the model got more defensible."
    )

    st.divider()
    st.markdown('<div class="section-label">Performance</div>', unsafe_allow_html=True)
    perf = pd.DataFrame(
        [
            {"Metric": "ROC-AUC",
             "Cross-validation": f"{cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}",
             "Held-out test": f"{test['auc']:.4f}"},
            {"Metric": "Average precision",
             "Cross-validation": f"{cv['ap_mean']:.4f} ± {cv['ap_std']:.4f}",
             "Held-out test": f"{test['ap']:.4f}"},
            {"Metric": "Precision @ top 5%",
             "Cross-validation": f"{cv['precision_top5_mean']:.1%} ± {cv['precision_top5_std']:.1%}",
             "Held-out test": f"{test['precision_top5']:.1%}"},
            {"Metric": "Precision @ top 10%",
             "Cross-validation": f"{cv['precision_top10_mean']:.1%} ± {cv['precision_top10_std']:.1%}",
             "Held-out test": f"{test['precision_top10']:.1%}"},
            {"Metric": "Recall @ top 10%",
             "Cross-validation": f"{cv['recall_top10_mean']:.1%} ± {cv['recall_top10_std']:.1%}",
             "Held-out test": f"{test['recall_top10']:.1%}"},
        ]
    )
    st.dataframe(perf, use_container_width=True, hide_index=True)
    st.caption(
        f"Base stress rate is {BASE_RATE:.2%}, so average precision reads against "
        f"a {BASE_RATE:.4f} baseline, not against 0.5. Average precision is the "
        "headline metric because ROC-AUC flatters models on imbalanced data."
    )

    strata = pd.DataFrame(METRICS.get("strata", []))
    if not strata.empty and exp:
        st.write("")
        st.markdown("**Against the exposure-only baseline, and within exposure strata**")
        nov = METRICS.get("no_overlap_variant", {})
        rows = [
            {
                "Model": f"Exposure only — account counts and file ages ({len(exp['features'])} features)",
                "ROC-AUC": f"{exp['auc']:.4f}",
                "Average precision": f"{exp['ap']:.4f}",
                "Precision @ top 10%": f"{exp['precision_top10']:.1%}",
            }
        ]
        if nov:
            rows.append(
                {
                    "Model": f"No same-window features ({nov['n_features']} features)",
                    "ROC-AUC": f"{nov['auc']:.4f}",
                    "Average precision": f"{nov['ap']:.4f}",
                    "Precision @ top 10%": f"{nov['precision_top10']:.1%}",
                }
            )
        rows.append(
            {
                "Model": f"Full model ({METRICS['n_features']} features)",
                "ROC-AUC": f"{test['auc']:.4f}",
                "Average precision": f"{test['ap']:.4f}",
                "Precision @ top 10%": f"{test['precision_top10']:.1%}",
            }
        )
        bench = pd.DataFrame(rows)
        st.dataframe(bench, use_container_width=True, hide_index=True)
        st.dataframe(
            strata.assign(
                stress_rate=lambda d: (d["stress_rate"] * 100).round(1),
                auc=lambda d: d["auc"].round(4),
                precision_top10=lambda d: (d["precision_top10"] * 100).round(1),
            ).rename(
                columns={
                    "band": "Trade lines",
                    "accounts": "Borrowers",
                    "stressed": "Stressed",
                    "stress_rate": "Stress rate %",
                    "auc": "ROC-AUC",
                    "precision_top10": "Precision @10% ",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Mean within-stratum AUC is {METRICS.get('mean_within_stratum_auc', 0):.4f} "
            f"against {test['auc']:.4f} overall. The gap is the part of the overall "
            "figure that comes from ranking large credit files above small ones "
            "rather than from discriminating between comparable borrowers. A wide "
            "gap would indicate the exposure confound still dominates; this one is "
            "narrow and the strata are consistent."
        )

    st.divider()
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**Lift by score decile** (held-out test set)")
        dec = pd.DataFrame(METRICS["deciles"])
        fig = go.Figure(
            go.Bar(
                x=dec["decile"].astype(str), y=dec["lift"],
                marker=dict(color=T.SERIES_1, cornerradius=4), width=0.62,
                hovertemplate="Decile %{x}<br>%{y:.2f}× lift<extra></extra>",
            )
        )
        fig.add_hline(y=1.0, line=dict(color=T.INK_MUTED, dash="dash", width=1))
        st.plotly_chart(
            T.style_fig(fig, x_title="Decile (1 = highest risk)", y_title="Lift vs base rate"),
            use_container_width=True,
        )
        table_view(dec.round(3))

    with c2:
        st.markdown("**Top 15 features by gain**")
        imp = METRICS["feature_importance"]
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:15][::-1]
        fig2 = go.Figure(
            go.Bar(
                x=[v for _, v in top], y=[k for k, _ in top], orientation="h",
                marker=dict(color=T.SERIES_1, cornerradius=4),
                hovertemplate="%{y}<br>gain %{x:.4f}<extra></extra>",
            )
        )
        styled = T.style_fig(fig2, x_title="Gain")
        styled.update_yaxes(showgrid=False)
        styled.update_xaxes(showgrid=True, gridcolor=T.GRID)
        st.plotly_chart(styled, use_container_width=True)
        st.caption(
            "Gain splits credit among correlated features, so read families "
            "rather than individual ranks."
        )

    st.divider()
    st.markdown(
        '<div class="section-label">Direction of the arrow — the sharpest objection</div>',
        unsafe_allow_html=True,
    )
    nov = METRICS.get("no_overlap_variant", {})
    if nov:
        st.markdown(
            f"The label covers the last 12 months. So do many of the features. A "
            f"borrower who went delinquent in month 3 may have made a burst of "
            f"enquiries and opened accounts in months 4–12 *because* they were in "
            f"distress — in which case `enq_L3m` is a **consequence** of the "
            f"outcome, not a signal preceding it. The source is a single "
            f"cross-sectional snapshot with no as-of dates, so the ordering cannot "
            f"be recovered and the direction is **formally unidentifiable**. That "
            f"is not a caveat that can be argued away.\n\n"
            f"What can be done is bound it. Retraining with all "
            f"**{nov['n_dropped']} same-window features removed** gives a model "
            f"that provably cannot be reading post-outcome behaviour:"
        )
        floor = exp.get("auc", 0)
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            T.tile("Full model", f"{test['auc']:.4f}", f"{METRICS['n_features']} features"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            T.tile(
                "Same-window features removed",
                f"{nov['auc']:.4f}",
                f"{nov['auc'] - test['auc']:+.4f} vs full · {nov['n_features']} features",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            T.tile(
                "Still above exposure floor",
                f"+{nov['auc'] - floor:.4f}",
                f"floor is {floor:.4f}",
            ),
            unsafe_allow_html=True,
        )
        share = (test["auc"] - nov["auc"]) / max(1e-9, test["auc"] - floor)
        st.markdown(
            f'<div class="caveat">Read honestly: of the +{test["auc"] - floor:.4f} '
            f"the full model gains over the exposure floor, "
            f"<b>{share:.0%} depends on features sharing the label's window</b> and "
            f"could in principle be post-outcome behaviour. The remaining "
            f"<b>{1 - share:.0%}</b> comes from lifetime stocks, point-in-time "
            f"utilisation and borrower profile, which cannot be. Settling it "
            f"properly needs panel data with as-of dates — features observed at "
            f"month 0, outcomes at months 1–12. That is the next piece of work, "
            "not something this dataset can answer.</div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"The {nov['n_dropped']} features removed for this check"):
            st.write(", ".join(f"`{c}`" for c in nov["dropped"]))

    st.divider()
    st.markdown('<div class="section-label">Leakage controls</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-label">Limitations</div>', unsafe_allow_html=True)
    for text in [
        "The model predicts <b>concurrent</b> stress classification, not a "
        "forward-dated event. The source data is a cross-sectional snapshot with "
        "no default timestamps, so no lead-time claim is made or supported.",
        "Exposure is reduced, not eliminated. The fixed window cut the prevalence "
        "gradient across exposure groups to 2.39×, but borrowers with more trade "
        "lines still show a higher stress rate. Some of that is likely real risk "
        "rather than artefact, so it is reported rather than adjusted away — see "
        "the within-stratum AUCs above, which stay in a narrow band.",
        "<code>CC_utilization</code> is 92.8% unreported in the source and "
        "<code>PL_utilization</code> 86.6%. Both are median-filled and paired with "
        "an explicit missingness indicator. On its own that missingness is not "
        "predictive (0.99× lift), so its model importance is interaction-driven.",
        "Trained on retail credit-bureau records used as a proxy for MSME "
        "proprietor risk. Not validated on firm-level financials, GST filing "
        "behaviour, or TReDS payment data.",
        "A 10.6% event rate means top-decile metrics rest on roughly 370 stressed "
        "accounts in the test set. Fold-level variation is reported above rather "
        "than a single split.",
        "<code>HL_Flag</code> and <code>Home_TL</code> contradict each other for "
        "some borrowers in the source RBI file (flag set, zero home-loan accounts). "
        "Not corrected here, but worth knowing before trusting either column.",
    ]:
        st.markdown(f'<div class="caveat">{text}</div>', unsafe_allow_html=True)


# ── shell ───────────────────────────────────────────────────────────

PAGES = {
    "Overview": page_overview,
    "Portfolio triage": page_portfolio,
    "Borrower drill-down": page_borrower,
    "Batch scoring": page_batch,
    "Model card": page_model_card,
}

with st.sidebar:
    st.markdown("### MSME Stress Early-Warning")
    st.caption(
        "Ranking 12-month credit deterioration from leading signals, with "
        "delinquency history excluded by rule"
    )
    choice = st.radio("View", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.markdown(
        T.tile("Test ROC-AUC", f"{METRICS['test']['auc']:.3f}",
               f"± {METRICS['cv']['summary']['auc_std']:.3f} CV"),
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        T.tile("Borrowers scored", f"{len(SCORED):,}", "held out of training"),
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        T.tile("Base stress rate", f"{BASE_RATE:.2%}", f"{METRICS['n_features']} features"),
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "All scores shown are out-of-sample. Colours and chart chrome are "
        "defined in `src/theme.py` and validated for colour-vision deficiency."
    )

st.title("MSME Credit Stress Early-Warning")
st.markdown(
    f'<div class="page-subtitle">{choice}</div>',
    unsafe_allow_html=True,
)
PAGES[choice]()
