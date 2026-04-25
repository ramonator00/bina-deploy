# =========================================================
# STANDARD LIBRARY
# =========================================================
import sqlite3
import pickle
from threading import Thread

# =========================================================
# DATA & VISUALIZATION
# =========================================================
import numpy as np
import pandas as pd
import plotly.express as px

# =========================================================
# DASH
# =========================================================
from dash import Dash, dcc, html, Input, Output

# =========================================================
# MACHINE LEARNING
# =========================================================
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# =========================================================
# LOAD DATA FROM SQLITE
# =========================================================
conn = sqlite3.connect("injury_analysis.db")
df = pd.read_sql("SELECT * FROM injury_data", conn)

# Labels
df["gender_label"] = df["gender"].map({"m": "Male", "f": "Female"}).fillna("Unknown")
df["injury_label"] = df["injury"].map({1: "Injury", 0: "No Injury"})

if "age" in df.columns:
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 100],
        labels=["<25", "25-35", "35-45", "45+"]
    )

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

default_x = "training_intensity" if "training_intensity" in numeric_cols else numeric_cols[0]
default_y = "load_score" if "load_score" in numeric_cols else numeric_cols[1]

# =========================================================
# LOAD MODEL (für Prediction Tab)
# =========================================================
with open("injury_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# =========================================================
# DASH APP
# =========================================================
app = Dash(__name__)

app.layout = html.Div([

    html.H2("Injury Analytics Dashboard"),

    dcc.Tabs([

        dcc.Tab(label="Dashboard", children=[

            html.Div([

                html.Div([
                    html.P(
                        "Interactive exploration of injury risk factors",
                        style={"color": "#6c757d"}
                    )
                ], style={"marginBottom": "20px"}),

                html.Div([

                    html.Div([
                        html.Label("Gender"),
                        dcc.Dropdown(
                            id="gender-filter",
                            options=[{"label": g, "value": g} for g in df["gender_label"].unique()],
                            value=list(df["gender_label"].unique()),
                            multi=True
                        )
                    ], style={"width": "20%", "display": "inline-block", "padding": "5px"}),

                    html.Div([
                        html.Label("Age Group"),
                        dcc.Dropdown(
                            id="age-filter",
                            options=[{"label": a, "value": a} for a in df["age_group"].dropna().unique()],
                            value=list(df["age_group"].dropna().unique()),
                            multi=True
                        )
                    ], style={"width": "20%", "display": "inline-block", "padding": "5px"}),

                    html.Div([
                        html.Label("Plot Type"),
                        dcc.Dropdown(
                            id="plot-type",
                            options=[
                                {"label": "Scatter", "value": "scatter"},
                                {"label": "Histogram", "value": "hist"},
                                {"label": "Box", "value": "box"},
                                {"label": "Violin", "value": "violin"},
                                {"label": "Density", "value": "density"},
                                {"label": "Correlation Heatmap", "value": "corr"},
                                {"label": "Line", "value": "line"},
                            ],
                            value="scatter"
                        )
                    ], style={"width": "20%", "display": "inline-block", "padding": "5px"}),

                    html.Div([
                        html.Label("X Axis"),
                        dcc.Dropdown(
                            id="x-axis",
                            options=[{"label": c, "value": c} for c in numeric_cols],
                            value=default_x
                        )
                    ], style={"width": "20%", "display": "inline-block", "padding": "5px"}),

                    html.Div([
                        html.Label("Y Axis"),
                        dcc.Dropdown(
                            id="y-axis",
                            options=[{"label": c, "value": c} for c in numeric_cols],
                            value=default_y
                        )
                    ], style={"width": "20%", "display": "inline-block", "padding": "5px"}),

                ], style={
                    "backgroundColor": "white",
                    "padding": "15px",
                    "borderRadius": "12px",
                    "boxShadow": "0px 4px 12px rgba(0,0,0,0.05)",
                    "marginBottom": "20px"
                }),

                html.Div([
                    html.Label("Features"),
                    dcc.Dropdown(
                        id="features",
                        options=[{"label": c, "value": c} for c in numeric_cols],
                        value=numeric_cols[:5],
                        multi=True
                    )
                ], style={
                    "backgroundColor": "white",
                    "padding": "15px",
                    "borderRadius": "12px",
                    "boxShadow": "0px 4px 12px rgba(0,0,0,0.05)",
                    "marginBottom": "20px"
                }),

                html.Div(id="kpis", style={"display": "flex", "gap": "15px", "marginBottom": "20px"}),
                html.Div(id="plots")

            ], style={
                "maxWidth": "1300px",
                "margin": "auto",
                "padding": "20px",
                "backgroundColor": "#f5f7fa",
                "fontFamily": "Arial"
            })

        ]),

        dcc.Tab(label="Prediction", children=[

            html.Div([

                html.H3("Injury Risk Prediction", style={"marginBottom": "20px"}),

                html.Div([

                    html.Button(
                        "Run Prediction",
                        id="predict-btn",
                        style={
                            "padding": "10px 20px",
                            "backgroundColor": "#4C78A8",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "6px",
                            "cursor": "pointer"
                        }
                    ),

                    html.Div(id="prediction-output"),
                    html.Div(id="feature-importance")

                ], style={"textAlign": "center"})

            ])

        ])

    ])

])

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)