"""Design tokens and chart chrome for the dashboard.

**Theme-agnostic by design.** Streamlit only reads `.streamlit/config.toml` from
the process's working directory, so launching the app from anywhere else drops
the pinned theme and Streamlit falls back to the viewer's OS preference. An
earlier version hardcoded near-black text, which vanished on a dark background.
Nothing here assumes a surface colour: every value was checked against both the
light chart surface (#fcfcfb) and Streamlit's default dark one (#0E1117).

Palettes, with the validator results that justified them — each passes on BOTH
surfaces, so one palette serves either theme:

  - categorical pair  #3987e5 / #d95926 — CVD ΔE 26.8, normal 31.8; passes light+dark
  - diverging poles   #3987e5 / #d03b3b — CVD ΔE 25.7, normal 31.9; passes light+dark

The diverging pair replaced an earlier red/green choice, which sat at CVD ΔE 9.1
under deuteranopia — technically above the floor but nearly 3× worse separation,
and red/green is the classic colour-vision failure case.

Text greys are mode-invariant, measured against both surfaces:

  - INK_EMPHASIS #767676 — 4.42:1 light / 4.16:1 dark
  - INK_MUTED    #898781 — 3.50:1 light / 5.26:1 dark (axis labels, UI-text tier)

Status colours (Low/Medium/High/Critical) are a fixed palette, never themed.
They clear 3:1 on the dark surface; on light, warning and serious fall below by
design, so the relief rule applies — tier names are always visible on the axis
and every chart has a table twin, so tier identity is never colour alone.

HTML text inherits Streamlit's resolved theme colour rather than setting its own,
and card surfaces use translucent greys, so both read correctly either way.
"""

from __future__ import annotations

import plotly.graph_objects as go

# ── categorical (identity) ──────────────────────────────────────────
SERIES_1 = "#3987e5"     # blue
SERIES_2 = "#d95926"     # orange

# ── diverging (polarity: raises vs lowers risk) ─────────────────────
POLE_POS = "#d03b3b"     # raises risk
POLE_NEG = SERIES_1      # lowers risk — same blue, so the two charts agree

# ── mode-invariant ink & chrome ─────────────────────────────────────
INK = "#767676"          # emphasis labels on charts
INK_2 = "#767676"        # chart base font
INK_MUTED = "#898781"    # axis ticks and titles
GRID = "rgba(137,135,129,0.28)"    # hairline gridline, reads on either surface
AXIS = "rgba(137,135,129,0.55)"    # baseline / reference rule

# Translucent so cards sit correctly on a light or dark plane.
CARD_BG = "rgba(127,127,127,0.06)"
CARD_BORDER = "rgba(127,127,127,0.22)"

# Note on ordinal ramps: a 5-step blue ramp validates on the light surface
# (#86b6ef → #104281), but 6 steps cannot clear the 0.06 adjacent-lightness gap
# given the light end must stay above 2:1. Where an ordered 6-band chart was
# needed, a single hue is used instead — bar length already encodes magnitude, so
# a ramp would double-encode it.

# ── status (state) — fixed, never themed ────────────────────────────
STATUS = {
    "Low": "#0ca30c",
    "Medium": "#fab219",
    "High": "#ec835a",
    "Critical": "#d03b3b",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def style_fig(
    fig: go.Figure,
    *,
    height: int = 320,
    show_legend: bool = False,
    x_title: str | None = None,
    y_title: str | None = None,
    y_range: list | None = None,
) -> go.Figure:
    """Apply the chart chrome once, so every figure reads as one system.

    Hairline solid gridlines (never dashed — dashing reads as a threshold),
    recessive axes, transparent plot background so the page theme shows through.
    """
    fig.update_layout(
        height=height,
        # Left margin has to hold the y-axis title plus its ticks, or the title
        # gets clipped by the plot edge.
        margin=dict(l=64, r=20, t=12, b=12),
        font=dict(family=FONT, size=12, color=INK_2),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            y=1.16,
            x=0,
            font=dict(size=11, color=INK_2),
            bgcolor="rgba(0,0,0,0)",
        ),
        # No explicit hover background: Plotly then derives it from the trace
        # colour, which stays legible under either theme.
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        bargap=0.45,
    )
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(size=11, color=INK_MUTED)) if x_title else None,
        showgrid=False,
        zeroline=False,
        linecolor=AXIS,
        linewidth=1,
        ticks="outside",
        ticklen=4,
        tickcolor=AXIS,
        tickfont=dict(size=11, color=INK_MUTED),
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(size=11, color=INK_MUTED)) if y_title else None,
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        griddash="solid",
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(size=11, color=INK_MUTED),
        range=y_range,
        # Bars must grow from a single zero baseline; a truncated axis overstates
        # differences. Percentages read against zero too, so this is safe for
        # the line charts as well.
        rangemode="tozero" if y_range is None else "normal",
    )
    return fig


# `color: inherit` throughout — Streamlit resolves the theme's text colour on
# <body>, so inheriting means these blocks are correct in light and dark without
# knowing which is active. Secondary text is dimmed with opacity rather than a
# fixed grey, for the same reason.
CSS = f"""
<style>
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1380px; }}

/* Hero — exactly one per view */
.hero-figure {{
    font-family: {FONT};
    font-size: 3.4rem;
    font-weight: 600;
    line-height: 1;
    letter-spacing: -0.02em;
    color: inherit;
}}
.hero-unit {{ font-size: 1.7rem; font-weight: 500; opacity: 0.55; margin-left: 0.06em; }}
.hero-caption {{
    font-size: 0.9rem; color: inherit; opacity: 0.68;
    margin-top: 0.45rem; max-width: 34rem; line-height: 1.5;
}}

/* Stat tile */
.tile {{
    border: 1px solid {CARD_BORDER};
    border-radius: 10px;
    padding: 0.85rem 1rem;
    background: {CARD_BG};
    height: 100%;
}}
.tile-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: inherit; opacity: 0.6;
    margin-bottom: 0.3rem;
}}
.tile-value {{ font-size: 1.5rem; font-weight: 600; color: inherit; line-height: 1.1; }}
.tile-delta {{ font-size: 0.76rem; color: inherit; opacity: 0.65; margin-top: 0.25rem; }}

/* Note strips */
.note {{
    border-left: 3px solid {SERIES_1};
    padding: 0.55rem 0.9rem;
    margin: 0.4rem 0 1.1rem 0;
    font-size: 0.88rem;
    color: inherit; opacity: 0.78;
}}
.caveat {{
    border-left: 3px solid {STATUS['Medium']};
    padding: 0.55rem 0.9rem;
    margin: 0.35rem 0;
    font-size: 0.86rem;
    color: inherit; opacity: 0.78;
}}
.rigour {{
    border: 1px solid {CARD_BORDER};
    border-radius: 10px;
    padding: 0.9rem 1rem;
    background: {CARD_BG};
    height: 100%;
}}
.rigour-title {{ font-size: 0.82rem; font-weight: 600; color: inherit; margin-bottom: 0.3rem; }}
.rigour-body {{ font-size: 0.8rem; color: inherit; opacity: 0.7; line-height: 1.45; }}

.section-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: inherit; opacity: 0.55;
    margin: 0.4rem 0 0.5rem 0;
}}
.page-subtitle {{
    color: inherit; opacity: 0.6;
    font-size: 0.94rem; margin: -0.5rem 0 1.4rem 0;
}}
h1 {{ font-weight: 600 !important; letter-spacing: -0.02em; }}
</style>
"""


def tile(label: str, value: str, delta: str | None = None) -> str:
    """Stat tile: label (sentence case) / value / optional delta."""
    d = f'<div class="tile-delta">{delta}</div>' if delta else ""
    return (
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div>{d}</div>'
    )


def rigour_card(title: str, body: str) -> str:
    return (
        f'<div class="rigour"><div class="rigour-title">{title}</div>'
        f'<div class="rigour-body">{body}</div></div>'
    )
