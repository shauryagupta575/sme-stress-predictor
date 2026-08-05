"""Stress label construction.

Kept separate from feature engineering so the label definition can be reviewed
on its own — it is the assumption the whole project rests on.
"""

import pandas as pd

from . import config as C


def build_stress_label(df: pd.DataFrame) -> pd.Series:
    """Return the binary stress label: RBI SMA-2 bucket or worse.

    Stressed (1) if the borrower has ever been 60+ days past due, or holds any
    trade line classified sub-standard / doubtful / loss.

    Sentinel values in the source (-99999 / -99998) mean "not reported", which
    for a delinquency count is an absence of recorded delinquency, so they are
    treated as zero rather than as missing.
    """
    dpd = df[C.DPD_60_COL].replace(C.SENTINEL_VALUES, 0).fillna(0)
    npa = (
        df[C.NPA_COUNT_COLS]
        .replace(C.SENTINEL_VALUES, 0)
        .fillna(0)
        .sum(axis=1)
    )
    return ((dpd > 0) | (npa > 0)).astype(int)


def label_report(df: pd.DataFrame, label: pd.Series) -> str:
    """Human-readable summary of the label, including the SMA-2/NPA split."""
    dpd = df[C.DPD_60_COL].replace(C.SENTINEL_VALUES, 0).fillna(0)
    npa = (
        df[C.NPA_COUNT_COLS].replace(C.SENTINEL_VALUES, 0).fillna(0).sum(axis=1)
    )
    dpd_only = ((dpd > 0) & (npa == 0)).sum()
    npa_only = ((npa > 0) & (dpd == 0)).sum()
    both = ((dpd > 0) & (npa > 0)).sum()

    lines = [
        "Stress label: RBI SMA-2 or worse (60+ DPD, or NPA-classified trade line)",
        f"  total rows        : {len(label):,}",
        f"  stressed          : {int(label.sum()):,} ({label.mean():.2%})",
        f"  60+ DPD only      : {dpd_only:,}",
        f"  NPA only          : {npa_only:,}",
        f"  both              : {both:,}",
    ]

    # Contrast against the original Approved_Flag-derived label, if present.
    if "Approved_Flag" in df.columns:
        old = (df["Approved_Flag"] != "P2").astype(int)
        agree = (old == label).mean()
        lines += [
            "",
            "Contrast with the original Approved_Flag label (P2=healthy):",
            f"  original stress rate : {old.mean():.2%}",
            f"  agreement with new   : {agree:.2%}",
            "  stress rate by Approved_Flag tier under the new label:",
        ]
        by_tier = label.groupby(df["Approved_Flag"]).mean().sort_index()
        for tier, rate in by_tier.items():
            lines.append(f"    {tier}: {rate:.2%}")
        lines.append(
            "  -> Approved_Flag barely separates genuine stress, which is why "
            "the original label could not support the project's claims."
        )

    return "\n".join(lines)
