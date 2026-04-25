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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# =========================================================
# GOOGLE COLAB
# =========================================================
from google.colab import drive, output

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

        # =========================================================
        # TAB 1: DEIN DASHBOARD (UNVERÄNDERT)
        # =========================================================
        dcc.Tab(label="Dashboard", children=[

            html.Div([

                # HEADER
                html.Div([
                    html.P("Interactive exploration of injury risk factors",
                           style={"color": "#6c757d"})
                ], style={"marginBottom": "20px"}),

                # FILTER PANEL
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

                # FEATURE SELECT
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

        # =========================================================
        # TAB 2: PREDICTION
        # =========================================================
        dcc.Tab(label="Prediction", children=[

            html.Div([

                html.H3("Injury Risk Prediction", style={"marginBottom": "20px"}),

                # INPUT CARD
                html.Div([

                    html.Div([
                        html.Label("Age"),
                        dcc.Slider(
                            id="age-input",
                            min=15, max=70, step=1, value=30,
                            marks={20: "20", 40: "40", 60: "60"}
                        )
                    ], style={"marginBottom": "20px"}),

                    html.Div([
                        html.Label("Training Intensity"),
                        dcc.Slider(
                            id="intensity-input",
                            min=1, max=5, step=1, value=3,
                            marks={1: "1", 3: "3", 5: "5"}
                        )
                    ], style={"marginBottom": "20px"}),

                    html.Div([
                        html.Label("Training Frequency"),
                        dcc.Input(
                            id="frequency-input",
                            type="number",
                            value=3,
                            style={"width": "100%"}
                        )
                    ], style={"marginBottom": "20px"}),

                    html.Div([
                        html.Label("Sleep Hours"),
                        dcc.Slider(
                            id="sleep-input",
                            min=3, max=10, step=0.5, value=7,
                            marks={4: "4h", 7: "7h", 9: "9h"}
                        )
                    ], style={"marginBottom": "20px"}),

                    html.Div([
                        html.Label("Stress Level"),
                        dcc.Slider(
                            id="stress-input",
                            min=1, max=5, step=1, value=3,
                            marks={1: "Low", 3: "Mid", 5: "High"}
                        )
                    ], style={"marginBottom": "20px"}),

                ], style={
                    "backgroundColor": "white",
                    "padding": "20px",
                    "borderRadius": "12px",
                    "boxShadow": "0px 4px 12px rgba(0,0,0,0.05)",
                    "marginBottom": "20px"
                }),

                # BUTTON + RESULT
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

html.Div(id="prediction-output", style={
    "marginTop": "20px",
    "fontSize": "22px",
    "fontWeight": "600"
}),

html.Div(id="feature-importance", style={
    "marginTop": "30px"
})

                ], style={"textAlign": "center"})

            ], style={
                "maxWidth": "700px",
                "margin": "auto",
                "padding": "20px"
            })

        ])

    ])

])

# =========================================================
# DASHBOARD CALLBACK (UNVERÄNDERT)
# =========================================================
@app.callback(
    Output("kpis", "children"),
    Output("plots", "children"),
    Input("gender-filter", "value"),
    Input("age-filter", "value"),
    Input("plot-type", "value"),
    Input("x-axis", "value"),
    Input("y-axis", "value"),
    Input("features", "value"),
)
def update_dashboard(genders, ages, plot_type, x, y, features):

    dff = df[
        (df["gender_label"].isin(genders)) &
        (df["age_group"].isin(ages))
    ].copy()

    if len(dff) > 1000:
        dff = dff.sample(1000, random_state=42)

    if dff.empty:
        return [], html.Div("No data available")

    def card(title, value):
        return html.Div([
            html.Div(title, style={"fontSize": "12px", "color": "#6c757d"}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "600"})
        ], style={
            "backgroundColor": "white",
            "padding": "15px",
            "borderRadius": "10px",
            "boxShadow": "0px 4px 12px rgba(0,0,0,0.05)",
            "minWidth": "120px"
        })

    kpis = [
        card("Rows", f"{len(dff):,}"),
        card("Injury Rate", f"{dff['injury'].mean()*100:.1f}%"),
        card("Avg X", f"{dff[x].mean():.2f}" if x in dff else "-"),
        card("Avg Y", f"{dff[y].mean():.2f}" if y in dff else "-"),
    ]

    if plot_type == "scatter":
        fig = px.scatter(dff, x=x, y=y, color="injury_label", symbol="gender_label")

    elif plot_type == "hist":
        fig = px.histogram(dff, x=x, color="injury_label")

    elif plot_type == "box":
        fig = px.box(dff, x="injury_label", y=x)

    elif plot_type == "violin":
        fig = px.violin(dff, x="injury_label", y=x)

    elif plot_type == "density":
        fig = px.density_contour(dff, x=x, y=y)

    elif plot_type == "line":
        fig = px.line(dff.sort_values(x), x=x, y=y)

    elif plot_type == "corr":
        fig = px.imshow(dff[features].corr())

    fig.update_layout(template="plotly_white", height=600)

    return kpis, dcc.Graph(figure=fig)

@app.callback(
    Output("prediction-output", "children"),
    Output("feature-importance", "children"),
    Input("predict-btn", "n_clicks"),
    Input("age-input", "value"),
    Input("intensity-input", "value"),
    Input("frequency-input", "value"),
    Input("sleep-input", "value"),
    Input("stress-input", "value"),
)

def predict(n, age, intensity, frequency, sleep, stress):

    if not n:
        return "", ""

    # =====================
    # BASELINE (WICHTIG)
    # =====================
    defaults = df.mean(numeric_only=True).to_dict()

    # =====================
    # USER INPUT
    # =====================
    defaults.update({
        "age": age,
        "training_intensity": intensity,
        "training_frequency": frequency,
        "sleep_hours": sleep,
        "stress_level": stress,
        "load_score": intensity * frequency if intensity and frequency else 0
    })

    if "gender" in feature_columns:
        defaults["gender"] = 0

    # =====================
    # MODEL INPUT
    # =====================
    input_df = pd.DataFrame([defaults])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # =====================
    # PREDICTION
    # =====================
    prob = model.predict_proba(input_df)[0][1]

    # =====================
    # FEATURE IMPORTANCE
    # =====================
    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False).head(8)

    fig = px.bar(
        importance,
        x=importance.values,
        y=importance.index,
        orientation="h"
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=20, t=30, b=30)
    )

    return (
        f"Injury Risk: {prob*100:.1f}%",
        dcc.Graph(figure=fig)
    )

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)