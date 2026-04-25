# =========================================================
# STANDARD LIBRARY
# =========================================================
import sqlite3
import pickle
import os

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
# LOAD DATA
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

conn = sqlite3.connect(os.path.join(BASE_DIR, "injury_analysis.db"))
df = pd.read_sql("SELECT * FROM injury_data", conn)

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
# LOAD MODEL
# =========================================================
with open(os.path.join(BASE_DIR, "injury_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "feature_columns.pkl"), "rb") as f:
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
                    html.P("Interactive exploration of injury risk factors",
                           style={"color": "#6c757d"})
                ], style={"marginBottom": "20px"}),

                html.Div([

                    dcc.Dropdown(
                        id="gender-filter",
                        options=[{"label": g, "value": g} for g in df["gender_label"].unique()],
                        value=list(df["gender_label"].unique()),
                        multi=True
                    ),

                    dcc.Dropdown(
                        id="age-filter",
                        options=[{"label": a, "value": a} for a in df["age_group"].dropna().unique()],
                        value=list(df["age_group"].dropna().unique()),
                        multi=True
                    ),

                    dcc.Dropdown(
                        id="plot-type",
                        options=[
                            {"label": "Scatter", "value": "scatter"},
                            {"label": "Histogram", "value": "hist"},
                            {"label": "Box", "value": "box"},
                            {"label": "Correlation", "value": "corr"},
                        ],
                        value="scatter"
                    ),

                    dcc.Dropdown(
                        id="x-axis",
                        options=[{"label": c, "value": c} for c in numeric_cols],
                        value=default_x
                    ),

                    dcc.Dropdown(
                        id="y-axis",
                        options=[{"label": c, "value": c} for c in numeric_cols],
                        value=default_y
                    ),

                ]),

                html.Div(id="kpis"),
                html.Div(id="plots")

            ])

        ]),

        dcc.Tab(label="Prediction", children=[

            html.Button("Run Prediction", id="predict-btn"),
            html.Div(id="prediction-output"),
            html.Div(id="feature-importance")

        ])

    ])

])

# =========================================================
# DASHBOARD CALLBACK
# =========================================================
@app.callback(
    Output("kpis", "children"),
    Output("plots", "children"),
    Input("gender-filter", "value"),
    Input("age-filter", "value"),
    Input("plot-type", "value"),
    Input("x-axis", "value"),
    Input("y-axis", "value"),
)
def update_dashboard(genders, ages, plot_type, x, y):

    dff = df[
        (df["gender_label"].isin(genders)) &
        (df["age_group"].isin(ages))
    ]

    if dff.empty:
        return "No data", ""

    kpis = f"Rows: {len(dff)} | Injury Rate: {dff['injury'].mean()*100:.1f}%"

    if plot_type == "scatter":
        fig = px.scatter(dff, x=x, y=y, color="injury_label")

    elif plot_type == "hist":
        fig = px.histogram(dff, x=x)

    elif plot_type == "box":
        fig = px.box(dff, x="injury_label", y=x)

    elif plot_type == "corr":
        fig = px.imshow(dff[numeric_cols].corr())

    return kpis, dcc.Graph(figure=fig)

# =========================================================
# PREDICTION CALLBACK
# =========================================================
@app.callback(
    Output("prediction-output", "children"),
    Output("feature-importance", "children"),
    Input("predict-btn", "n_clicks"),
)
def predict(n):

    if not n:
        return "", ""

    input_df = df[feature_columns].mean().to_frame().T

    prob = model.predict_proba(input_df)[0][1]

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False).head(8)

    fig = px.bar(importance, x=importance.values, y=importance.index, orientation="h")

    return f"Injury Risk: {prob*100:.1f}%", dcc.Graph(figure=fig)

# =========================================================
# RUN
# =========================================================
server = app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)