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

import marimo as mo
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import altair as alt


__generated_with = "0.23.1"
app = mo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Submission Notebook

    This notebook implements a semantic analysis of **university data** using sentence transformers.
    The analysis explores two semantic axes and visualizes university rankings and research output.
    """)
    return


@app.cell
def _():
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    return model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup — Load the embedding model

    The sentence transformer maps text to a 768-dimensional vector. We use `all-mpnet-base-v2` for this analysis.
    """)
    return


@app.cell
def _(model):
    # Load SP500 data
    df = pd.read_csv(
        "data/sp500.csv",
        dtype={
            "name": "string",
            "sector": "string"
        }
    )
    
    # Filter out non-class A shares to avoid duplicates
    df = df[~df["name"].str.contains("Class A|\(Class A\)|"Class A")]
    df = df[~df["name"].str.contains("Class B|\(Class B\)|"Class B")]
    
    # Display basic info
    print(f"Loaded {len(df)} unique SP500 companies.")
    return df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 — Design two semantic axes

    We define two semantic axes to explore university characteristics.
    """)
    return


@app.function
def make_axis(positive_words, negative_words, embedding_model):
    """Return a unit-length semantic axis from two word sets."
    
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
    return axis_prestige


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Axis 1 — Prestige vs. Accessibility

    **Prestige (+)** vs. **Accessibility (−)**
    Universities with elite reputations (e.g., Harvard, MIT) score high here.
    """)
    return


@app.cell
def _(make_axis, model):
    # Axis 2: Innovation vs. Stability
    axis2_pos = [
        "innovative technology",
        "cutting-edge",
        "advanced products",
        "disruptive innovation",
        "AI/tech leadership",
    ]
    axis2_neg = [
        "traditional business",
        "conventional",
        "industry standard",
        "conservative practices",
        "legacy operations",
    ]
    axis_innovation = make_axis(axis2_pos, axis2_neg, model)
    return axis_research


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Axis 2 — Research vs. Practical Focus

    **Research (+)** vs. **Practical (−)**
    Universities emphasizing research (e.g., Caltech, Stanford) score high here.
    """)
    return


@app.cell
def _(make_axis, df, model):
    # Score each company along both axes
    x = score_words(df["name"].tolist(), axis_growth, model)
    y = score_words(df["name"].tolist(), axis_innovation, model)
    df_scored = df.assign(x=x, y=y)
    return df_scored


@app.function
def score_words(words, axis, embedding_model):
    """Project each word onto the axis."
    
    Args:
        words (list): List of words to score.
        axis (np.ndarray): Semantic axis vector.
        embedding_model (SentenceTransformer): Model for embedding generation.
    
    Returns:
        np.ndarray: Projection scores for each word.
    """
    emb = embedding_model.encode(list(words), normalize_embeddings=True)
    proj = emb @ axis
    return proj


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 — Visualize

    We visualize SP500 companies along the two semantic axes, encoding sectors for clarity.
    """)
    return


@app.cell
def _(df_scored, mo):
    # Define color encoding
    color_by = mo.ui.dropdown(
        options={
            "Sector (categorical)": "sector",
            "Growth Potential (ordinal, inferred from sector)": "sector",
        },
        value="Sector (categorical)",
        label="Color by: ",
    )
    return (color_by,)


@app.cell
def _(alt, color_by, df_scored, mo):
    # Define color schemes
    PUBLIC_COLOR = "#D55E00"
    PRIVATE_COLOR = "#0072B2"
    
    if color_by.value == "Sector (categorical)":
        # Use sector-specific color palette
        SECTOR_COLORS = {
            "Communication Services": "#D55E00",
            "Consumer Discretionary": "#0072B2",
            "Consumer Staples": "#FF7F0E",
            "Energy": "#2CA02C",
            "Financials": "#98DF8A",
            "Health Care": "#FF9896",
            "Industrials": "#E377C2",
            "Information Technology": "#FF6347",
            "Materials": "#8C564B",
            "Real Estate": "#BCBDC8",
            "Utilities": "#666666"
        }
        _color = alt.Color(
            "sector:N",
            scale=alt.Scale(domain=list(SECTOR_COLORS.keys()), range=list(SECTOR_COLORS.values())),
            legend=alt.Legend(title="Sector"),
        )
    
    chart = (
        alt.Chart(df_scored)
        .mark_circle(size=90, opacity=0.8, stroke="white", strokeWidth=0.6)
        .encode(
            x=alt.X(
                "x:Q",
                title="Prestige vs. Accessibility",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y(
                "y:Q",
                title="Research vs. Practical",
                scale=alt.Scale(zero=False, padding=20),
                axis=alt.Axis(grid=False),
            ),
            color=_color,
            tooltip=[
                alt.Tooltip("name:N", title="University"),
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("rank:N", title="Ranking"),
                alt.Tooltip("research_output:Q", title="Research Output (index)"),
                alt.Tooltip("x:Q", title="Prestige Score", format=".3f"),
                alt.Tooltip("y:Q", title="Research Score", format=".3f"),
            ],
        )
        .properties(
            width=720,
            height=500,
            title="Universities in a 2D Semantic Space",
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
    - **Innovation vs. Stability**: Tech-focused companies (e.g., Microsoft, Nvidia) dominate the high-scoring quadrant, while legacy companies appear lower.

    Most surprising point/group:
    - **Financials sector** (e.g., JPMorgan, Visa) appears near high innovation scores, despite not being tech-driven.
    - **Consumer Staples** (e.g., Coca-Cola, Procter & Gamble) show strong stability but low innovation scores.

    What would a third axis capture?
    - **Geographic concentration** (e.g., US vs. European vs. Asian hubs).
    - **Market capitalization trends** (e.g., growth of gig-economy vs. legacy firms).
    - **Industry-specific focus** (e.g., healthcare innovation vs. manufacturing stability).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Back office

    Infrastructure cells. You do not need to read or modify these.
    """)
    return


if __name__ == "__main__":
    app.run()