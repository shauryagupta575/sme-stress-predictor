"""Stress label construction.

Kept separate from feature engineering so the label definition can be reviewed
on its own — it is the assumption the whole project rests on. See config.py for
why the observation window is fixed rather than lifetime.
"""

import numpy as np
import pandas as pd

from . import config as C


def build_stress_label(df: pd.DataFrame) -> pd.Series:
    """Return the binary stress label: deterioration within the last 12 months.

    Stressed (1) if, in the last 12 months, any trade line went delinquent or
    was classified sub-standard / doubtful / loss.

    Sentinel values in the source (-99999 / -99998) mean "not reported", which
    for a delinquency count is an absence of recorded delinquency, so they are
    treated as zero rather than as missing.
    """
    deliq = df[C.DELIQ_12M_COL].replace(C.SENTINEL_VALUES, 0).fillna(0)
    npa = (
        df[C.NPA_12M_COLS]
        .replace(C.SENTINEL_VALUES, 0)
        .fillna(0)
        .sum(axis=1)
    )
    return ((deliq > 0) | (npa > 0)).astype(int)


def has_label_columns(df: pd.DataFrame) -> bool:
    """Whether `df` carries the source columns needed to derive the true label.

    Callers should use this rather than checking config column names themselves.
    An earlier version of the dashboard inlined those names, so renaming the
    label columns broke batch scoring with an AttributeError that only fired on
    upload — the one path least likely to be exercised in testing.
    """
    return C.DELIQ_12M_COL in df.columns and all(
        c in df.columns for c in C.NPA_12M_COLS
    )


def exposure_band(df: pd.DataFrame) -> pd.Series:
    """Bucket borrowers by number of trade lines held.

    Used to report metrics within comparable-exposure strata, so the model's
    performance can be read separately from the fact that borrowers with more
    accounts are riskier.
    """
    return pd.cut(
        df["Total_TL"].fillna(0),
        bins=C.EXPOSURE_BANDS,
        labels=C.EXPOSURE_BAND_LABELS,
    )


def label_report(df: pd.DataFrame, label: pd.Series) -> str:
    """Human-readable summary, including the exposure gradient the fixed window
    was chosen to reduce."""
    deliq = df[C.DELIQ_12M_COL].replace(C.SENTINEL_VALUES, 0).fillna(0)
    npa = (
        df[C.NPA_12M_COLS].replace(C.SENTINEL_VALUES, 0).fillna(0).sum(axis=1)
    )
    deliq_only = ((deliq > 0) & (npa == 0)).sum()
    npa_only = ((npa > 0) & (deliq == 0)).sum()
    both = ((deliq > 0) & (npa > 0)).sum()

    lines = [
        "Stress label: deterioration within a fixed 12-month observation window",
        "  (any delinquency, or any NPA classification, in the last 12 months)",
        f"  total rows          : {len(label):,}",
        f"  stressed            : {int(label.sum()):,} ({label.mean():.2%})",
        f"  delinquency only    : {deliq_only:,}",
        f"  NPA classified only : {npa_only:,}",
        f"  both                : {both:,}",
    ]

    # The exposure gradient: how much the label rises purely with account count.
    band = exposure_band(df)
    grad = label.groupby(band, observed=True).mean()
    if len(grad) > 1:
        lines += [
            "",
            "  Prevalence by number of trade lines held:",
            "    " + "   ".join(f"{k}: {v:.1%}" for k, v in grad.items()),
            f"    gradient (highest band / lowest band): {grad.iloc[-1] / grad.iloc[0]:.2f}x",
            "    (the lifetime label this replaced ran at 5.45x)",
        ]

    return "\n".join(lines)


def build_exposure_matrix(df: pd.DataFrame, medians: pd.Series | None = None):
    """Feature matrix for the exposure-only baseline model.

    Account counts and credit-file ages only. Whatever AUC this reaches is the
    floor the full model has to clear to be worth anything — a far more honest
    benchmark than 0.5.
    """
    X = df[C.EXPOSURE_COLS].replace(C.SENTINEL_VALUES, np.nan)
    X = X.assign(age_span=X["Age_Oldest_TL"] - X["Age_Newest_TL"])
    if medians is None:
        medians = X.median()
    return X.fillna(medians), medians
