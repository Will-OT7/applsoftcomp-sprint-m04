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
import numpy as np
import pandas as pd
import altair as alt
from sentence_transformers import SentenceTransformer

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Submission Notebook

    This notebook implements a semantic analysis of **university data** using sentence transformers.
    The analysis explores two semantic axes and visualizes university rankings and research output.
    """)
    return


@app.cell
def _(SentenceTransformer):
    model = SentenceTransformer("all-mpnet-base-v2")
    model
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup — Load the embedding model

    The sentence transformer maps text to a 768-dimensional vector. We use `all-mpnet-base-v2` for this analysis.
    """)
    return


@app.cell
def _(model):
    # Load university data
    df = pd.read_csv(
        "data/universities.csv",
        dtype={
            "name": "string",
            "country": "string",
            "rank": "Int64",
            "research_output": "float64",
            "international_student_ratio": "float64",
            "faculty_student_ratio": "float64",
            "public": "bool",
        }
    )
    
    # Display basic info
    print(f"Loaded {len(df)} universities.")
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
    # Axis 1: Prestige vs. Accessibility
    axis1_pos = [
        "elite university",
        "top global ranking",
        "prestigious institution",
        "world-class research",
        "high academic reputation",
    ]
    axis1_neg = [
        "community college",
        "accessible university",
        "local institution",
        "open admissions",
        "affordable education",
    ]
    axis_prestige = make_axis(axis1_pos, axis1_neg, model)
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
    # Axis 2: Research vs. Practical Focus
    axis2_pos = [
        "cutting-edge research",
        "high research output",
        "academic excellence",
        "innovation hub",
        "faculty research",
    ]
    axis2_neg = [
        "practical education",
        "vocational training",
        "applied skills",
        "industry partnerships",
        "practical focus",
    ]
    axis_research = make_axis(axis2_pos, axis2_neg, model)
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
    # Score each university along both axes
    x = score_words(df["name"].tolist(), axis_prestige, model)
    y = score_words(df["name"].tolist(), axis_research, model)
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

    We visualize universities along the two semantic axes, encoding attributes for clarity.
    """)
    return


@app.cell
def _(df_scored, mo):
    # Define color encoding
    color_by = mo.ui.dropdown(
        options={
            "Ranking (ordinal)": "rank",
            "International Student Ratio (ordinal)": "international_student_ratio",
            "Faculty-Student Ratio (ordinal)": "faculty_student_ratio",
            "Public Status (categorical)": "public",
        },
        value="Ranking (ordinal)",
        label="Color by: ",
    )
    return (color_by,)


@app.cell
def _(alt, color_by, df_scored, mo):
    # Define color schemes
    PUBLIC_COLOR = "#D55E00"
    PRIVATE_COLOR = "#0072B2"
    
    if color_by.value == "Ranking (ordinal)":
        # Ranked by QS/ARWU (higher = better)
        _color = alt.Color(
            "rank:N",
            scale=alt.Scale(domain=["High", "Very High", "High", "Very High 1", "Very High 2", "Very High 3", "High 1", "High 2", "High 3", "Medium", "Low", "None"],
                          range=["#0072B2", "#2CA02C", "#FF7F0E", "#D55E00", "#98DF8A", "#FF9896", "#8C564B", "#E377C2", "#7F7F7F", "#BCBDC8", "#666666", "#333333"]),
            legend=alt.Legend(title="Ranking Category"),
        )
    elif color_by.value == "International Student Ratio (ordinal)":
        _color = alt.Color(
            "international_student_ratio:Q",
            scale=alt.Scale(scheme="viridis", type="sequential", domain=[0, 100]),
            legend=alt.Legend(title="% International Students"),
        )
    elif color_by.value == "Faculty-Student Ratio (ordinal)":
        _color = alt.Color(
            "faculty_student_ratio:Q",
            scale=alt.Scale(scheme="inferno", domain=[1, 20]),
            legend=alt.Legend(title="Faculty-Student Ratio"),
        )
    elif color_by.value == "Public Status (categorical)":
        _color = alt.Color(
            "public:N",
            scale=alt.Scale(domain=[True, False], range=[PRIVATE_COLOR, PUBLIC_COLOR]),
            legend=alt.Legend(title="Public University"),
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
    - **Prestige vs. Accessibility**: Elite institutions cluster at high scores, while community colleges cluster at low scores.
    - **Research vs. Practical**: Research-focused universities (e.g., Caltech, MIT) dominate the high-scoring quadrant, while vocational schools appear low.

    Most surprising point/group:
    - **Asian institutions** (e.g., Tsinghua, Peking University) appear near the high-prestige, high-research quadrant, despite not being top-ranked globally by traditional metrics.
    - **Public universities in Europe** (e.g., Heidelberg, Oxford) show a mix of high prestige and practical focus.

    What would a third axis capture?
    - **Geographic spread**: How universities are distributed globally (e.g., North America vs. Asia vs. Europe).
    - **Historical legacy**: Institutions with long histories (e.g., Cambridge, Oxford) might cluster differently.
    - **Industry alignment**: How closely universities align with local industries (e.g., engineering schools near tech hubs).
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