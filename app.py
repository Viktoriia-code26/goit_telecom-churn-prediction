from dash import Dash, html, dcc, Input, Output, State
from numpy import isfinite
import pandas as pd
from pathlib import Path
import joblib
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
)

BASE = Path(__file__).resolve().parent

LABELS = {
    "is_tv_subscriber": "TV Subscription",
    "is_movie_package_subscriber": "Movie Package",
    "subscription_age": "Time as a Customer",
    "bill_avg": "Average Bill",
    "service_failure_count": "Service Issues",
    "download_avg": "Average Download Usage",
    "upload_avg": "Average Upload Usage",
    "download_over_limit": "Download Limit Exceedances",
    "reamining_contract": "Time Left on Contract",
    "remaining_contract_missing": "Contract Unknown",
    "actual_churn": "Actual Outcome",
    "predicted_churn": "Predicted Outcome",
    "predicted_probability": "Churn Risk",
}

try:
    model = joblib.load(BASE / "models" / "churn_model.pkl")
    model_error = None
except Exception as error:
    model = None
    model_error = str(error)


def style_figure(figure, height=350):
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=45, r=24, t=58, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", color="#344054", size=12),
        title=dict(font=dict(size=16, color="#101828")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    figure.update_xaxes(
        gridcolor="#edf1f7",
        zeroline=False,
        linecolor="#dfe5ee",
    )
    figure.update_yaxes(
        gridcolor="#edf1f7",
        zeroline=False,
        linecolor="#dfe5ee",
    )
    return figure


def graph_card(figure, class_name="chart-card"):
    return html.Div(
        dcc.Graph(
            figure=style_figure(figure),
            responsive=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            style={"width": "100%"},
        ),
        className=f"panel {class_name}",
    )


def display_value(column, value):
    if pd.isna(value):
        return "—"
    if column == "predicted_probability":
        return f"{float(value):.1%}"
    if column in ["actual_churn", "predicted_churn"]:
        return "Churn" if value == 1 else "Retained"
    if column in ["is_tv_subscriber", "is_movie_package_subscriber", "remaining_contract_missing"]:
        return "Yes" if value == 1 else "No"
    if column in ["subscription_age", "reamining_contract", "download_avg", "upload_avg"]:
        return f"{float(value):.2f}"
    if column in ["bill_avg", "service_failure_count", "download_over_limit"]:
        return f"{value:,.0f}"
    return str(value)


def metric_card(label, value, note=None, tone="blue"):
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
        html.Div(note or "Model evaluation", className=f"metric-note metric-{tone}"),
    ], className="metric-card")


def analytics():
    try:
        results = pd.read_csv(BASE / "data" / "predictions.csv")
        columns = ["actual_churn", "predicted_churn", "predicted_probability"]
        results[columns] = results[columns].apply(pd.to_numeric, errors="raise")

        if results.empty or results[columns].isna().any().any():
            raise ValueError("CSV is empty or contains missing labels/probabilities.")
        if not results[columns[:2]].isin([0, 1]).all().all():
            raise ValueError("Actual and predicted labels must be 0 or 1.")
        if not results["predicted_probability"].between(0, 1).all():
            raise ValueError("Probabilities must be between 0 and 1.")

    except Exception as error:
        return html.Div(
            f"Cannot load analytics: {error}",
            className="error-box",
        )

    actual = results["actual_churn"]
    predicted = results["predicted_churn"]
    probability = results["predicted_probability"]
    both_classes = actual.nunique() == 2

    scores = {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1 score": f1_score(actual, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(actual, probability) if both_classes else None,
    }

    churn_share = float(actual.mean())
    high_risk_share = float((probability >= 0.7).mean())

    metrics = html.Div([
        metric_card("Customers evaluated", f"{len(results):,}", "Saved predictions", "blue"),
        metric_card("Actual churn rate", f"{churn_share:.1%}", "Share of churned customers", "red"),
        metric_card("Accuracy", f"{scores['Accuracy']:.3f}", "Correct classifications", "green"),
        metric_card("F1 score", f"{scores['F1 score']:.3f}", "Balance of precision and recall", "purple"),
        metric_card("High-risk customers", f"{high_risk_share:.1%}", "Predicted risk ≥ 70%", "orange"),
    ], className="metrics-grid")

    # Confusion matrix
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    fig_matrix = go.Figure(go.Heatmap(
        z=matrix,
        x=["Retained", "Churn"],
        y=["Retained", "Churn"],
        colorscale=[
            [0.0, "#eef4ff"],
            [0.5, "#94b7ff"],
            [1.0, "#315ddc"],
        ],
        text=matrix,
        texttemplate="%{text}",
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Customers: %{z}<extra></extra>",
    ))
    fig_matrix.update_layout(
        title="Actual vs. Predicted Outcomes",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    fig_matrix.update_yaxes(autorange="reversed")

    # Risk distribution
    fig_risk = go.Figure()
    for label, name, color in [
        (0, "Retained", "#5276e8"),
        (1, "Churn", "#ef8868"),
    ]:
        fig_risk.add_trace(go.Histogram(
            x=probability[actual == label],
            name=name,
            marker_color=color,
            xbins=dict(start=0, end=1, size=0.05),
            opacity=0.72,
        ))
    fig_risk.update_layout(
        title="Distribution of Churn Risk",
        barmode="overlay",
        xaxis_title="Predicted probability",
        yaxis_title="Customers",
    )
    fig_risk.update_xaxes(range=[0, 1], tickformat=".0%")

    charts = [
        html.Div([
            graph_card(fig_risk, "chart-wide"),
            graph_card(fig_matrix, "chart-narrow"),
        ], className="charts-row")
    ]

    # ROC
    if both_classes:
        fpr, tpr, _ = roc_curve(actual, probability)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC = {scores['ROC-AUC']:.3f}",
            line=dict(color="#315ddc", width=3),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(dash="dash", color="#98a2b3"),
        ))
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
        )
    else:
        fig_roc = None

    # Contract risk
    fig_contract = None
    if {"reamining_contract", "remaining_contract_missing"}.issubset(results.columns):
        duration = pd.to_numeric(results["reamining_contract"], errors="coerce")
        missing = pd.to_numeric(results["remaining_contract_missing"], errors="coerce")
        invalid = ~missing.isin([0, 1]) | (missing.eq(0) & (~isfinite(duration) | duration.lt(0)))

        if not invalid.any():
            groups = pd.cut(
                duration,
                bins=[0, .25, .5, 1, 2, float("inf")],
                labels=["0–3 months", "3–6 months", "6–12 months", "1–2 years", "2+ years"],
                include_lowest=True,
            ).cat.add_categories(["Unknown"])
            groups.loc[missing.eq(1)] = "Unknown"

            summary = results.groupby(groups, observed=True)["predicted_probability"].agg(["mean", "count"])

            fig_contract = go.Figure(go.Bar(
                x=summary.index.astype(str),
                y=summary["mean"],
                customdata=summary["count"],
                marker_color="#5276e8",
                hovertemplate="%{x}<br>Mean risk: %{y:.1%}<br>Customers: %{customdata}<extra></extra>",
            ))
            fig_contract.update_layout(
                title="Churn Risk by Time Left on Contract",
                yaxis_title="Mean churn probability",
                xaxis_title="",
            )
            fig_contract.update_yaxes(range=[0, 1], tickformat=".0%")

    # Feature importance
    fig_importance = None
    if model is not None:
        try:
            estimator = model.steps[-1][1] if hasattr(model, "steps") else model
            names = model.feature_names_in_

            if hasattr(model, "steps") and len(model.steps) > 1:
                names = model[:-1].get_feature_names_out()

            importance = pd.Series(
                estimator.feature_importances_,
                index=names,
            ).sort_values().tail(10)

            fig_importance = go.Figure(go.Bar(
                x=importance.values,
                y=[
                    LABELS.get(str(name).split("__", 1)[-1], str(name))
                    for name in importance.index
                ],
                orientation="h",
                marker_color="#5276e8",
            ))
            fig_importance.update_layout(
                title="Feature Importance",
                xaxis_title="Importance",
                yaxis_title="",
            )
            fig_importance.update_yaxes(automargin=True)
        except (AttributeError, ValueError, TypeError):
            pass

    second_row = []
    if fig_importance is not None:
        second_row.append(graph_card(fig_importance, "chart-card"))
    if fig_contract is not None:
        second_row.append(graph_card(fig_contract, "chart-card"))
    if fig_roc is not None:
        second_row.append(graph_card(fig_roc, "chart-card"))

    if second_row:
        charts.append(html.Div(second_row, className="charts-grid-3"))

    # Insights based only on displayed data
    insight_items = [
        ("Model quality", f"Accuracy is {scores['Accuracy']:.1%} and F1 is {scores['F1 score']:.1%}."),
        ("Customer risk", f"{high_risk_share:.1%} of evaluated customers have predicted churn risk of at least 70%."),
        ("Observed churn", f"Actual churn in the saved evaluation set is {churn_share:.1%}."),
    ]

    insights = html.Div([
        html.Div([
            html.Div("Key insights", className="section-title"),
            html.Div("A compact summary of the current evaluation set.", className="section-subtitle"),
        ], className="section-heading"),
        html.Div([
            html.Div([
                html.Div(str(i + 1), className="insight-number"),
                html.Div([
                    html.Div(title, className="insight-title"),
                    html.Div(text, className="insight-text"),
                ]),
            ], className="insight-item")
            for i, (title, text) in enumerate(insight_items)
        ], className="insights-list"),
    ], className="panel insights-panel")

    preview = results.head(20)
    cell_style = {
        "padding": "12px 16px",
        "textAlign": "left",
        "whiteSpace": "nowrap",
        "borderBottom": "1px solid #e7ecf3",
    }

    table = html.Details([
        html.Summary("View prediction results", className="table-summary"),
        html.P(
            f"Showing {len(preview)} of {len(results):,} rows. "
            "Incorrect predictions are highlighted in red.",
            className="table-note",
        ),
        html.Div(
            html.Table([
                html.Thead(html.Tr([
                    html.Th(
                        LABELS.get(c, c),
                        title=c,
                        scope="col",
                        style={
                            **cell_style,
                            "background": "#f3f6fb",
                            "position": "sticky",
                            "top": "0",
                        },
                    )
                    for c in results.columns
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            display_value(c, value),
                            style={
                                **cell_style,
                                **(
                                    {"color": "#b42318", "fontWeight": "600"}
                                    if c == "predicted_churn"
                                    and row["actual_churn"] != row["predicted_churn"]
                                    else {}
                                ),
                            },
                        )
                        for c, value in row.items()
                    ])
                    for _, row in preview.iterrows()
                ]),
            ], className="results-table"),
            className="table-scroll",
        ),
    ], className="panel results-details")

    return html.Div([
        html.Div([
            html.Div("Model Performance", className="section-title"),
            html.Div(
                "Evaluation metrics and model diagnostics from saved predictions.",
                className="section-subtitle",
            ),
        ], className="section-heading"),
        metrics,
        *charts,
        insights,
        table,
    ], className="analytics-section")


app = Dash(__name__, assets_folder=str(BASE / "assets"), assets_url_path="/assets")
server = app.server


def field(name, value, step=1, yes_no=False):
    label = LABELS[name]
    if name == "reamining_contract":
        label += " (years; 0 means no time left)"
    if name in ["download_avg", "upload_avg"]:
        label += " (leave blank if unknown)"

    if yes_no:
        control = dcc.Dropdown(
            id=name,
            options=[
                {"label": "No", "value": 0},
                {"label": "Yes", "value": 1},
            ],
            value=value,
            clearable=False,
            className="form-control dash-dropdown",
        )
    else:
        control = dcc.Input(
            id=name,
            type="number",
            min=0,
            max=4500,
            value=value,
            step=step,
            className="form-control",
        )

    return html.Div([
        html.Label(label, htmlFor=name, className="field-label"),
        control,
    ], className="field-group")


sidebar = html.Aside([
    html.Div([
        html.Div("ML", className="brand-mark"),
        html.Div([
            html.Div("Churn", className="brand-title"),
            html.Div("Analytics", className="brand-subtitle"),
        ]),
    ], className="brand"),

    html.Div([
        html.Div("Prediction input", className="sidebar-title"),
        html.Div(
            "Enter customer values and calculate churn risk.",
            className="sidebar-copy",
        ),
    ]),

    html.Div([
        field("is_tv_subscriber", 0, yes_no=True),
        field("is_movie_package_subscriber", 0, yes_no=True),
        field("subscription_age", 2.0, step=0.01),
        field("bill_avg", 50),
        field("reamining_contract", 1.0, step=0.01),
        field("service_failure_count", 0),
        field("download_avg", 50.0, step=0.01),
        field("upload_avg", 10.0, step=0.01),
        field("download_over_limit", 0),
        
        dcc.Checklist(id="contract-unknown", options=[{"label": "Contract duration unknown", "value": "unknown"}], className="sidebar-copy"),
    ], className="form-stack"),

    html.Button(
        "Predict churn risk",
        id="prediction-button",
        n_clicks=0,
        className="primary-button",
    ),

    html.Div(
        id="prediction-result",
        **{"aria-live": "polite"},
        className="prediction-result",
    ),
], className="sidebar")


app.layout = html.Div([
    sidebar,

    html.Main([
        html.Div([
            html.Div([
                html.Div("CUSTOMER RETENTION", className="eyebrow"),
                html.H1("Customer Churn Prediction", className="page-title"),
                html.P(
                    "Explore model performance, churn risk and customer-level predictions.",
                    className="page-subtitle",
                ),
            ]),
            html.Div([
                html.Div("Model status", className="status-label"),
                html.Div(
                    "Ready" if model is not None else "Unavailable",
                    className=f"status-pill {'status-ready' if model is not None else 'status-error'}",
                ),
            ], className="status-box"),
        ], className="page-header"),

        analytics(),
    ], className="main-content"),
], className="app-shell")

@app.callback(
        Output("reamining-contract", "disabled"), 
        Input("contract-unknown", "value")
        )
def disable_unknown_contract(selection):
    return "unknown" in (selection or [])

@app.callback(
    Output("prediction-result", "children"),
    Input("prediction-button", "n_clicks"),
    State("is_tv_subscriber", "value"),
    State("is_movie_package_subscriber", "value"),
    State("subscription_age", "value"),
    State("bill_avg", "value"),
    State("service_failure_count", "value"),
    State("download_avg", "value"),
    State("upload_avg", "value"),
    State("download_over_limit", "value"),
    State("reamining_contract", "value"),
    State("contract-unknown", "value"),
    prevent_initial_call=True,
)
def predict_churn(
    n_clicks,
    is_tv_subscriber,
    is_movie_package_subscriber,
    subscription_age,
    bill_avg,
    service_failure_count,
    download_avg,
    upload_avg,
    download_over_limit,
    reamining_contract,
    contract_unknown,
):
    if "unknown" in (contract_unknown or []):
        reamining_contract = None

    if model is None:
        return html.Div(
            f"Model unavailable: {model_error}",
            className="prediction-error",
        )

    values = [
        is_movie_package_subscriber,
        is_tv_subscriber,
        subscription_age,
        bill_avg,
        service_failure_count,
        download_over_limit,
    ]

    if any(value is None for value in values):
        return html.Div(
            "Fill in all fields required fields. Contract duration, download and upload may be unknown.",
            className="prediction-error",
        )

    numbers = values + [v for v in [download_avg, upload_avg, reamining_contract] if v is not None] 

    if any(
        not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
        for value in numbers
    ):
        return html.Div(
            "Enter finite, non-negative numbers.",
            className="prediction-error",
        )

    if any(
        value not in (0, 1)
        for value in [is_tv_subscriber, is_movie_package_subscriber]
    ):
        return html.Div(
            "Choose Yes or No for each subscription.",
            className="prediction-error",
        )

    if service_failure_count != int(service_failure_count):
        return html.Div(
            "Service failures must be a whole number.",
            className="prediction-error",
        )

    if download_over_limit != int(download_over_limit):
        return html.Div(
            "Download over limit must be a whole number.",
            className="prediction-error",
        )

    remaining_contract_missing = int(reamining_contract is None)

    float_columns = [
        "subscription_age",
        "reamining_contract",
        "download_avg",
        "upload_avg",
    ]

    int_columns = [
        "is_tv_subscriber",
        "is_movie_package_subscriber",
        "bill_avg",
        "service_failure_count",
        "download_over_limit",
        "remaining_contract_missing",
    ]

    if bill_avg != int(bill_avg):
        return html.Div(
            "Average Bill must be a whole number.",
            className="prediction-error",
        )

    new_client = pd.DataFrame([{
        "is_tv_subscriber": int(is_tv_subscriber),
        "is_movie_package_subscriber": int(is_movie_package_subscriber),
        "subscription_age": subscription_age,
        "bill_avg": bill_avg,
        "service_failure_count": int(service_failure_count),
        "download_avg": float("nan") if download_avg is None else download_avg,
        "upload_avg": float("nan") if upload_avg is None else upload_avg,
        "download_over_limit": int(download_over_limit),
        "reamining_contract": 0.0 if remaining_contract_missing else reamining_contract,
        "remaining_contract_missing": remaining_contract_missing,
    }])

    new_client = new_client.astype({
        **{column: "float64" for column in float_columns},
        **{column: "int64" for column in int_columns},
    })

    try:
        new_client = new_client.loc[:, model.feature_names_in_]
        classes = list(model.classes_)

        if len(classes) != 2 or set(classes) != {0, 1}:
            raise ValueError("Expected classes 0 = retained, 1 = churn.")

        probability = float(
            model.predict_proba(new_client)[0, classes.index(1)]
        )

        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability.")

        prediction = model.predict(new_client)[0]

    except Exception as error:
        return html.Div(
            f"Prediction failed: {error}",
            className="prediction-error",
        )

    if probability >= 0.7:
        risk = "High risk"
        risk_class = "risk-high"
    elif probability >= 0.4:
        risk = "Medium risk"
        risk_class = "risk-medium"
    else:
        risk = "Low risk"
        risk_class = "risk-low"

    return html.Div([
        html.Div("Churn risk", className="result-label"),
        html.Div(f"{probability:.1%}", className=f"result-value {risk_class}"),
        html.Div(risk, className=f"result-risk {risk_class}"),
        html.Progress(
            value=str(probability),
            max="1",
            className=f"risk-progress {risk_class}",
        ),
        html.Div(
            f"Model prediction: {'Churn' if prediction == 1 else 'Retained'}",
            className="result-prediction",
        ),
        html.Div(
            "Risk bands: low < 40%, medium 40–70%, high ≥ 70%.",
            className="result-note",
        ),
    ], className="result-card")


if __name__ == "__main__":
    app.run(debug=True)
