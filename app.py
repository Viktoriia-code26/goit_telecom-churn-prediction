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

# 1. Загружаем сохранённую модель один раз при запуске.
try:
    model = joblib.load(BASE / "models" / "churn_model.pkl")
    model_error = None
except Exception as error:
    model = None
    model_error = str(error)

# 2. Общий вид графиков и формат чисел.
def chart(figure):
    figure.update_layout(
        template="plotly_white",
        margin=dict(l=45, r=25, t=60, b=45),
        font=dict(family="Arial", color="#26334d"),
    )
    return dcc.Graph(figure=figure, responsive=True,
                     config={"displaylogo": False}, style={"width": "100%", "minWidth": 0})

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



# 3. Метрики и графики по сохранённому CSV.
def analytics():
    # Графики используют сохранённые результаты, а не данные из формы
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
        return html.P(f"Cannot load analytics: {error}", style={"color": "#b42318"})

    actual = results["actual_churn"]
    predicted = results["predicted_churn"]
    probability = results["predicted_probability"]
    both_classes = actual.nunique() == 2
    scores = {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(actual, probability) if both_classes else None,
    }
    content = [
        html.H2("Model Performance"),
        html.P(f"{len(results):,} saved predictions. Undefined precision, recall and F1 are shown as 0."),
        html.Div([
            html.Div([
              html.Div(name),
              html.H2(f"{value:.3f}" if value is not None else "N/A"),  
            ], style={"background": "#eef3ff", "padding": "18px", "borderRadius": "12px"})
            for name, value in scores.items()
            ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(130px, 1fr))", "gap": "12px"}),]
    
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    
    figure = go.Figure(go.Heatmap(
        z=matrix, x=["Retained", "Churn"], y=["Retained", "Churn"],
        colorscale="Blues", text=matrix, texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>"
        "Predicted: %{x}<br>"
        "Customers: %{z}<extra></extra>",
    ))
    figure.update_layout(title="Actual vs. Predicted Outcomes", xaxis_title="Predicted", yaxis_title="Actual")
    figure.update_yaxes(autorange="reversed")
    content.append(chart(figure))

    figure = go.Figure()
    for label, name, color in [(0, "Retained", "#5276e8"), (1, "Churn", "#ef8868")]:
        figure.add_trace(go.Histogram(
            x=probability[actual == label], name=name, marker_color=color,
            xbins=dict(start=0, end=1, size=0.05), opacity=0.7,
        ))

    figure.update_layout(title="Distribution of Churn Risk", barmode="overlay", xaxis_title="Probability", yaxis_title="Customers")
    figure.update_xaxes(range=[0, 1], tickformat=".0%")
    content.append(chart(figure))

    if both_classes:
        fpr, tpr, _ = roc_curve(actual, probability)
        figure = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {scores['ROC-AUC']:.3f}"))
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                    name="Random baseline"))
        figure.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        content.append(chart(figure))
    else:
        content.append(html.P("ROC curve and ROC-AUC require both actual classes."))

    if {"reamining_contract", "remaining_contract_missing"}.issubset(results.columns):
        duration = pd.to_numeric(results["reamining_contract"], errors="coerce")
        missing = pd.to_numeric(results["remaining_contract_missing"], errors="coerce")
        invalid = ~missing.isin([0, 1]) | (missing.eq(0) & (~isfinite(duration) | duration.lt(0)))
        if invalid.any():
            content.append(html.P("Contract chart unavailable: invalid durations or missing flags."))
        else:
            groups = pd.cut(duration, bins=[0, .25, .5, 1, 2, float("inf")],
                            labels=["0-3 months", "3-6 months", "6-12 months", "1-2 years", "2+ years"],
                            include_lowest=True).cat.add_categories(["Unknown"])
            groups.loc[missing.eq(1)] = "Unknown"
            summary = results.groupby(groups, observed=True)["predicted_probability"].agg(["mean", "count"])
            figure = go.Figure(go.Bar(x=summary.index.astype(str), y=summary["mean"],
                                      customdata=summary["count"], marker_color="#5276e8",
                                      hovertemplate="%{x}<br>Mean risk: %{y:.1%}<br>Customers: %{customdata}<extra></extra>"))
            figure.update_layout(title="Churn Risk by Time Left on Contract", yaxis_title="Mean churn probability")
            figure.update_yaxes(range=[0, 1], tickformat=".0%")
            content.append(chart(figure))
    else:
        content.append(html.P("Contract chart needs reamining_contract and remaining_contract_missing columns."))

    if model is not None:
        try:
            estimator = model.steps[-1][1] if hasattr(model, "steps") else model
            names = model.feature_names_in_
            if hasattr(model, "steps") and len(model.steps) > 1:
                names = model[:-1].get_feature_names_out()
            importance = pd.Series(estimator.feature_importances_, index=names).sort_values().tail(15)
            figure = go.Figure(go.Bar(x=importance.values, y=[LABELS.get(str(name).split("__", 1)[-1], str(name)) for name in importance.index], orientation="h", marker_color="#5276e8"))
            figure.update_layout(title="Feature Importance", xaxis_title="Importance")
            figure.update_yaxes(automargin=True)
            content.append(chart(figure))
        except (AttributeError, ValueError, TypeError):
            content.append(html.P("Feature importance is unavailable for this model."))

    # Меняем только отображение: исходные данные для метрик остаются прежними.
    preview = results.head(20)
    cell_style = {"padding": "12px 16px", "textAlign": "left", "whiteSpace": "nowrap", "borderBottom": "1px solid #e7ecf3"}
    content.append(html.Details([
        html.Summary("View Prediction Results", style={"fontWeight": "600", "cursor": "pointer", "padding": "16px 0"}),
        html.P(f"Showing {len(preview)} of {len(results):,} rows. — means missing data. Subscription, download and upload units match the training dataset."),
        html.Div(html.Table([
            html.Thead(html.Tr([
                html.Th(LABELS.get(c, c), title=c, scope="col", style={**cell_style, "background": "#edf2fc", "position": "sticky", "top": "0"})
                for c in results.columns
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(display_value(c, value), style={
                        **cell_style,
                        **({"color": "#b42318", "fontWeight": "600"} if c == "predicted_churn" and row["actual_churn"] != row["predicted_churn"] else {}),
                    }) for c, value in row.items()
                ], style={"background": "#ffffff" if i % 2 == 0 else "#f7f9fc"})
                for i, (_, row) in enumerate(preview.iterrows())
            ]),
        ], style={"borderCollapse": "collapse", "width": "100%", "fontSize": "14px"}),
            style={"width": "100%", "maxWidth": "100%", "minWidth": 0,
                   "overflowX": "auto", "overflowY": "auto", "maxHeight": "440px", "border": "1px solid #e0e6ef", "borderRadius": "12px"}),
        html.P("Incorrect predictions are highlighted in red."),
    ], style={"width": "100%", "minWidth": 0, "maxWidth": "100%"}))
    return html.Div(content, style={"width": "100%", "minWidth": 0, "maxWidth": "100%"})


# 4. Простая вертикальная форма.
app = Dash(__name__)
server = app.server  # Используется Gunicorn в Docker: app:server.


def field(name, value, step=1, yes_no=False):
    label = LABELS[name]
    if name == "reamining_contract":
        label += " (Years; leave blank if unknown)"
    if yes_no:
        control = dcc.Dropdown(
            id=name, options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
            value=value, clearable=False,
        )
    else:
        # Обе границы нужны для кнопок +/− в Dash 4.4.1.
        # Максимум 900 сохранён из вашей формы; это не ограничение модели.
        control = dcc.Input(id=name, type="number", min=0, max=4500,
                            value=value, step=step, style={"width": "100%"})
    return html.Div([
        html.Label(label, htmlFor=name, style={"display": "block", "marginBottom": "6px"}),
        control,
    ])


app.layout = html.Div([
    html.H1("Churn Prediction"),
    html.P("Enter customer information to predict churn."),
    html.H3("Customer Details"),
    html.P("Use the same units as in the training data."),
    field("is_tv_subscriber", 0, yes_no=True),
    field("is_movie_package_subscriber", 0, yes_no=True),
    field("subscription_age", 2.0, step=0.1),
    field("bill_avg", 50),
    field("service_failure_count", 0),
    field("download_avg", 50.0, step=0.1),
    field("upload_avg", 10.0, step=0.1),
    field("download_over_limit", 0),
    field("reamining_contract", 1.0, step=0.1),
    html.Button("Predict Churn Risk", id="prediction-button", n_clicks=0,
                style={"padding": "12px", "background": "#395fd4", "color": "white",
                       "border": "none", "borderRadius": "6px", "cursor": "pointer", "fontSize": "16px"}),
    html.Div(id="prediction-result", **{"aria-live": "polite"}),
    html.Hr(),
    analytics(),
], style={"width": "100%", "maxWidth": "1000px", "boxSizing": "border-box",
          "margin": "32px auto", "padding": "24px", "minWidth": 0,
          "fontFamily": "Arial, sans-serif", "display": "flex",
          "flexDirection": "column", "gap": "12px"})


# 5. Единственный callback: прогноз по нажатию кнопки.

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
    prevent_initial_call=True
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
    reamining_contract):
    if model is None:
        return html.P(f"Model unavailable: {model_error}", style={"color": "#b42318"})
    values = [
        is_movie_package_subscriber,
        is_tv_subscriber,
        subscription_age,
        bill_avg,
        service_failure_count,
        download_avg,
        upload_avg,
        download_over_limit,
    ]
    if any(value is None for value in values):
        return "Fill in all fields except the optional contract duration."

    numbers = values + ([] if reamining_contract is None else [reamining_contract])

    if any(not isinstance(value, (int, float)) or not isfinite(value) or value < 0 for value in numbers):
        return "Enter finite, non-negative numbers."

    if any(value not in (0, 1) for value in [is_tv_subscriber, is_movie_package_subscriber]):
        return "Choose Yes or No for each subscription."

    if service_failure_count != int(service_failure_count):
        return "Service failures must be a whole number."

    if download_over_limit != int(download_over_limit):
        return "Download over limit must be a whole number."

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
        return "Average Bill must be a whole number."
    
    new_client = pd.DataFrame([{
        "is_tv_subscriber": int(is_tv_subscriber),
        "is_movie_package_subscriber": int(is_movie_package_subscriber),
        "subscription_age": subscription_age,
        "bill_avg": bill_avg,
        "service_failure_count": int(service_failure_count),
        "download_avg": download_avg,
        "upload_avg": upload_avg,
        "download_over_limit": int(download_over_limit),
        "reamining_contract": 0.0 if remaining_contract_missing else reamining_contract,
        "remaining_contract_missing": remaining_contract_missing,
    }])
    new_client = new_client.astype({
        **{column: "float64" for column in float_columns},
        **{column: "int64" for column in int_columns},
    })

    try:
        # Используем имена и порядок признаков из обученной модели.
        new_client = new_client.loc[:, model.feature_names_in_]
        classes = list(model.classes_)
        if len(classes) != 2 or set(classes) != {0, 1}:
            raise ValueError("Expected classes 0 = retained, 1 = churn.")
        probability = float(model.predict_proba(new_client)[0, classes.index(1)])
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability.")
        prediction = model.predict(new_client)[0]
    except Exception as error:
        return html.P(f"Prediction failed: {error}", style={"color": "#b42318"})

    if probability >= 0.7:
        risk, color = "High risk", "#b42318"
    elif probability >= 0.4:
        risk, color = "Medium risk", "#9a6700"
    else:
        risk, color = "Low risk", "#18794e"

    return html.Div([
        html.P("Churn Risk"),
        html.H2(f"{probability:.1%}", style={"color": color, "fontSize": "42px"}),
        html.P(risk, style={"color": color}),
        html.Progress(value=str(probability), max="1", style={"width": "100%", "accentColor": color}),
        html.P(f"Model prediction: {'Churn' if prediction == 1 else 'Retained'}"),
        html.Small("Display bands: low < 40%, medium 40-70%, high ≥ 70%. They may differ from the model classification threshold."),
        html.P("Result uses the values submitted with the button. Submit again after editing."),
    ], style={"padding": "24px", "background": "#f3f6fc", "borderRadius": "12px"})


if __name__ == "__main__":
    app.run(debug=True)