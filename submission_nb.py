# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.10.0",
#     "sentence-transformers>=2.7.0",
#     "numpy>=1.24",
#     "pandas>=2.0",
#     "matplotlib>=3.7",
#     "scipy>=1.11",
#     "ipython>=8.0",
#     "anywidget>=0.9",
#     "seaborn==0.13.2",
#     "altair==6.0.0"
# ]
# ///

import marimo


__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # S&P 500 Semantic Axes

    This notebook implements a semantic analysis of **S&P 500 company names** using sentence transformers.
    It follows the same pipeline as the worked example: define two semantic axes, score every term, and visualize the resulting 2D map.
    """)
    return


@app.cell
def _(SentenceTransformer):
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup — Load the embedding model

    The sentence transformer maps text to a 768-dimensional vector. We use `all-mpnet-base-v2` for this analysis.
    """)
    return


@app.cell
def _(pd):
    # Load SP500 data
    df = pd.read_csv(
        "data/sp500.csv",
        dtype={
            "name": "string",
            "sector": "category",
        },
    )

    # Keep one share class per company where the sample includes multiple listings.
    df = (
        df[~df["name"].str.contains(r"\(Class B\)|Class B", regex=True, na=False)]
        .assign(name=lambda x: x["name"].str.replace(r"\(Class A\)|Class A", "", regex=True).str.strip())
        .drop_duplicates(subset=["name", "sector"])
        .reset_index(drop=True)
    )

    # Display basic info
    print(f"Loaded {len(df)} unique SP500 companies.")
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 — Design two semantic axes

    We define two semantic axes to explore how company names are positioned in embedding space:

    - **Horizontal** — stable / legacy (−) ↔ growth / disruption (+)
    - **Vertical** — physical assets / infrastructure (−) ↔ digital technology / software (+)
    """)
    return


@app.cell
def _(np):
    def make_axis(positive_words, negative_words, embedding_model):
        """Return a unit-length semantic axis from two word sets.

        Args:
            positive_words (list): Words representing the positive pole.
            negative_words (list): Words representing the negative pole.
            embedding_model (SentenceTransformer): Model for embedding generation.

        Returns:
            np.ndarray: Unit-length semantic axis vector.
        """
        pos_emb = embedding_model.encode(positive_words, normalize_embeddings=True)
        neg_emb = embedding_model.encode(negative_words, normalize_embeddings=True)
        pole_pos = pos_emb.mean(axis=0)
        pole_neg = neg_emb.mean(axis=0)
        v = pole_pos - pole_neg
        return v / (np.linalg.norm(v) + 1e-10)

    return (make_axis,)


@app.cell
def _():
    def score_words(words, axis, embedding_model):
        """Project each word onto the axis."""
        emb = embedding_model.encode(list(words), normalize_embeddings=True)
        return emb @ axis

    return (score_words,)


@app.cell
def _(make_axis, model):
    # Axis 1: Growth vs. Stability
    axis1_pos = [
        "fast-growing",
        "innovative",
        "high-tech",
        "market leader",
        "disruptive",
    ]
    axis1_neg = [
        "stable",
        "traditional",
        "conservative",
        "blue-chip",
        "legacy",
    ]
    axis_growth = make_axis(axis1_pos, axis1_neg, model)
    return (axis_growth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Axis 1 — Stability vs. Growth

    **Growth / disruption (+)** vs. **stable / legacy (−)**
    This axis should pull companies with technology, platform, or expansion-oriented associations away from older, steadier business language.
    """)
    return


@app.cell
def _(make_axis, model):
    # Axis 2: Digital technology vs. physical assets
    axis2_pos = [
        "software platform",
        "cloud computing",
        "artificial intelligence",
        "digital services",
        "internet technology",
    ]
    axis2_neg = [
        "industrial equipment",
        "physical infrastructure",
        "raw materials",
        "manufacturing plants",
        "real estate assets",
    ]
    axis_digital = make_axis(axis2_pos, axis2_neg, model)
    return (axis_digital,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Axis 2 — Physical Assets vs. Digital Technology

    **Digital technology / software (+)** vs. **physical assets / infrastructure (−)**
    This axis contrasts firms whose names evoke information technology with firms whose names evoke tangible infrastructure, materials, and property.
    """)
    return


@app.cell
def _(axis_digital, axis_growth, df, model, score_words):
    # Score each company along both axes
    x = score_words(df["name"].tolist(), axis_growth, model)
    y = score_words(df["name"].tolist(), axis_digital, model)
    df_scored = df.assign(x=x, y=y)
    df_scored.head()
    return (df_scored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 — Visualize

    We visualize SP500 companies along the two semantic axes, encoding sectors for clarity.
    """)
    return


@app.cell
def _(mo):
    # Define color encoding
    color_by = mo.ui.dropdown(
        options={
            "Sector (categorical)": "sector",
        },
        value="sector",
        label="Color by: ",
    )
    return (color_by,)


@app.cell
def _(alt, color_by, df_scored, mo):
    # Okabe-Ito-inspired categorical palette, extended for the 11 sectors.
    SECTOR_COLORS = {
        "Communication Services": "#0072B2",
        "Consumer Discretionary": "#D55E00",
        "Consumer Staples": "#CC79A7",
        "Energy": "#000000",
        "Financials": "#E69F00",
        "Health Care": "#56B4E9",
        "Industrials": "#999999",
        "Information Technology": "#009E73",
        "Materials": "#F0E442",
        "Real Estate": "#8A63A8",
        "Utilities": "#6B6B6B",
    }

    _color = alt.Color(
        f"{color_by.value}:N",
        scale=alt.Scale(
            domain=list(SECTOR_COLORS.keys()),
            range=list(SECTOR_COLORS.values()),
        ),
        legend=alt.Legend(title="Sector"),
    )

    chart = (
        alt.Chart(df_scored)
        .mark_circle(size=90, opacity=0.8, stroke="white", strokeWidth=0.6)
        .encode(
            x=alt.X(
                "x:Q",
                title="stable / legacy ←     → growth / disruption",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y(
                "y:Q",
                title="physical assets ←     → digital technology",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(grid=False),
            ),
            color=_color,
            tooltip=[
                alt.Tooltip("name:N", title="Company"),
                alt.Tooltip("sector:N", title="Sector"),
                alt.Tooltip("x:Q", title="growth score", format=".3f"),
                alt.Tooltip("y:Q", title="digital score", format=".3f"),
            ],
        )
        .properties(
            width=720,
            height=500,
            title="S&P 500 companies in a 2D semantic space",
        )
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, titleFontSize=12)
        .interactive()
    )
    
    mo.vstack([color_by, chart])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observations

    What separates along each axis?
    - **Growth vs. Stability**: Companies in Communication Services and Tech sectors cluster at high scores, while traditional sectors (e.g., Utilities) cluster at low scores.
    - **Digital Technology vs. Physical Assets**: Tech-focused companies (e.g., Microsoft, Nvidia) should score high, while industrials, materials, real estate, utilities, and energy companies should lean lower.

    Most surprising point/group:
    - Financial services and payment companies may appear near the digital/innovation side because names such as Visa and Mastercard are often discussed with digital networks, even though their sector is not Information Technology.
    - Consumer Staples companies can look semantically stable because their names are tied to mature household brands rather than fast-growth language.

    What would a third axis capture?
    - A third axis could capture consumer-facing brands vs. business-to-business firms, which cuts across sectors and is not the same as growth or digital intensity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Back office

    Infrastructure cells. You do not need to read or modify these.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import pandas as pd
    import altair as alt

    return SentenceTransformer, alt, mo, np, pd


if __name__ == "__main__":
    app.run()
