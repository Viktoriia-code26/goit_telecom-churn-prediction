# Telecom Customer Churn Prediction

Machine learning project for predicting customer churn for a
telecommunications company.
The project includes exploratory data analysis, data preprocessing,
model selection and tuning, evaluation, model persistence, and a
Dash application for predicting churn risk for a new customer.

## Project Objective

The goal of the project is to build a classification model that predicts
whether a telecommunications customer is likely to churn (`churn = 1`)
based on subscription, billing, contract, service usage, and service
failure information.

The final application allows a user to enter information for a new
customer and receive:

-   a churn prediction;
-   the predicted probability of churn;
-   a visual summary of model performance;
-   additional analysis of churn risk.

## Dataset

The modeling notebook uses the dataset `internet_service_churn.csv`.

The original dataset contains **72,274 observations and 11 columns**,
including the target variable `churn`.

Dataset columns and engineered feature:

| Column | Description |
|---|---|
| `is_tv_subscriber` | TV subscription flag (0/1) |
| `is_movie_package_subscriber` | Movie package subscription flag (0/1) |
| `subscription_age` | Subscription duration in the dataset's units |
| `bill_avg` | Average bill amount |
| `reamining_contract` | Remaining contract duration; the app assumes years |
| `service_failure_count` | Number of service failures |
| `download_avg` | Average download measure; confirm units from the dataset source |
| `upload_avg` | Average upload measure; confirm units from the dataset source |
| `download_over_limit` | Integer measure of download-limit exceedances, not a binary flag |
| `remaining_contract_missing` | Engineered flag: 1 if contract duration was missing |
| `churn` | Target: 1 = churned, 0 = retained; excluded from model inputs |

The model uses ten input features, including the engineered missing-value flag.

The `id` column is excluded from model training.

> **Note:** `reamining_contract` keeps the spelling used in the original
> dataset and trained model.

## Exploratory Data Analysis

The analysis includes:

-   dataset structure and descriptive statistics;
-   missing-value analysis;
-   duplicate detection;
-   validation of numerical values;
-   correlation analysis;
-   churn-class distribution and numerical-feature histograms;
-   churn analysis related to missing contract information;
-   feature-importance analysis for the final Random Forest model.

No duplicated rows were found.

One invalid negative value (`-0.02`) was detected in `subscription_age`.
Since subscription duration cannot be negative, it was replaced with
`0`.

### Missing Values

Missing values were found in:

-   `reamining_contract`: **21,572** missing values;
-   `download_avg`: **381** missing values;
-   `upload_avg`: **381** missing values.

Missing contract information was found to be strongly associated with
churn in this dataset:

-   customers with available `reamining_contract`: churn rate ≈
    **40.1%**;
-   customers with missing `reamining_contract`: churn rate ≈ **91.4%**.

Because the missingness itself contains useful predictive information,
an additional binary feature was created:

``` python
remaining_contract_missing
```

It indicates whether `reamining_contract` was originally missing.

## Data Preprocessing

The data was split into training and test sets using an **80/20
stratified split** with `random_state=42`.

Resulting sizes:

-   training set: **57,819** observations;
-   test set: **14,455** observations.

Preprocessing is performed inside a Scikit-learn `Pipeline` /
`ColumnTransformer` to reduce the risk of data leakage.

The preprocessing strategy is:

-   `download_avg` and `upload_avg` → missing values replaced with the
    training-set mean;
-   `reamining_contract` → missing values replaced with `0`;
-   `remaining_contract_missing` → preserves information about
    originally missing contract values;
-   remaining features → passed through without imputation;
-   Logistic Regression additionally uses `StandardScaler`.

## Models

Two classification algorithms were evaluated:

1.  **Logistic Regression**
2.  **Random Forest Classifier**

Hyperparameters were selected using **GridSearchCV with 5-fold
cross-validation**, optimizing the **F1-score**.

### Logistic Regression

The following parameter grid was tested:

``` python
{
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l1", "l2"]
}
```

Best parameters:

``` text
C = 0.1
penalty = l2
```

Best cross-validation F1-score:

``` text
0.8926
```

### Random Forest

The following parameter grid was tested:

``` python
{
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}
```

Best parameters:

``` text
n_estimators = 100
max_depth = None
min_samples_split = 5
min_samples_leaf = 1
```

Best cross-validation F1-score:

``` text
0.9466
```

## Model Performance

Performance on the held-out test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.876 | 0.875 | 0.907 | 0.890 | 0.931 |
| Random Forest | **0.941** | **0.957** | **0.935** | **0.946** | **0.981** |

For the Random Forest model, the confusion matrix was:

``` text
[[6107,  338],
 [ 518, 7492]]
```

The model correctly identified **7,492 of 8,010** customers who churned
in the test set.

Rows of the confusion matrix are actual classes; columns are predicted classes,
both ordered as 0 (retained), 1 (churned). There were 518 missed churn cases
and 338 false alarms.

Random Forest achieved the higher cross-validation F1-score and also
outperformed Logistic Regression on all reported test metrics. It is the
final model used by the application.

## Feature Importance

Random Forest feature importance showed that the most influential
feature was `reamining_contract`.

The largest feature importances were approximately:

| Feature | Importance |
|---|---:|
| `reamining_contract` | 0.572 |
| `download_avg` | 0.120 |
| `remaining_contract_missing` | 0.086 |
| `upload_avg` | 0.077 |
| `subscription_age` | 0.055 |
| `bill_avg` | 0.045 |

These values describe the fitted model's impurity-based feature importance;
they do not establish causal effects or measure the improvement from adding a feature.

## Dash

The Dash application is implemented in `app.py`.

It allows the user to enter customer information and returns:

-   predicted churn class;
-   churn probability;
-   a low, medium or high churn-risk message.

Display bands are low below 40%, medium from 40% to below 70%, and high
from 70%. These illustrative bands are separate from the model
classification and are not calibrated business thresholds.

Leave the contract field blank when its duration is unknown. After editing
inputs, click **Predict Churn Risk** to update the result. Predictions made
in the form are not appended to `predictions.csv`; the charts use saved
test-set results.

The application also displays:

-   prediction results;
-   Accuracy, Precision, Recall, F1 and ROC-AUC;
-   confusion matrix;
-   distribution of predicted churn probabilities;
-   ROC curve;
-   average predicted churn probability by remaining contract duration;
-   feature importance.

## Project Structure

```text
goit_telecom-churn-prediction/
├── data/
│   ├── internet_service_churn.csv
│   └── predictions.csv
├── models/
│   └── churn_model.pkl
├── notebooks/
│   └── Project_eda_train_modeling.ipynb
├── app.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

``` bash
git clone https://github.com/Viktoriia-code26/goit_telecom-churn-prediction.git
cd goit_telecom-churn-prediction
```

Create and activate a virtual environment:

``` bash
python3 -m venv .venv
```

On Windows, use `python -m venv .venv` if your Python command is `python`.

macOS / Linux:

``` bash
source .venv/bin/activate
```

Windows (Command Prompt):

``` bash
.venv\Scripts\activate
```

Install dependencies:

``` bash
python -m pip install -r requirements.txt
```

## Run the Dash Application

Start the application with:

``` bash
python app.py
```

Then open [http://localhost:8050](http://localhost:8050).
The local command uses Dash debug mode; Docker uses Gunicorn.

## Saved Model

The final trained Random Forest pipeline is saved as:

``` text
models/churn_model.pkl
```

The application loads the model with `joblib` and uses the same trained
preprocessing pipeline for inference.

## Prediction Example

The modeling notebook includes an example customer for which the final
model produced:

``` text
Churn probability: 17.1%
Prediction: Low churn risk
```

## Technologies

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Dash and Plotly
-   Gunicorn
-   Joblib

## Run with Docker

The application can also be run inside a Docker container.

### Build the Docker image

From the project root directory:

```bash
docker build -t churn-app .
```

### Run the container

```bash
docker run --rm -p 127.0.0.1:8050:8050 churn-app
```

Open the application in your browser:

```text
http://localhost:8050
```

Docker installs runtime dependencies during the build and serves the app
with Gunicorn on port 8050 as a non-root user. Docker must be running.

Stop the foreground container with `Ctrl+C`. If port 8050 is occupied, stop
the old container or use `-p 127.0.0.1:8051:8050` and open port 8051.
Rebuild the image and recreate the container after changing the code,
saved model, or CSV files.

## Reproduce the Analysis

Use Python 3.12 and install the runtime requirements, then the notebook tools:

```bash
python -m pip install ipykernel matplotlib seaborn
```

Open `notebooks/Project_eda_train_modeling.ipynb`, select the project's
Python environment, and run all cells from a fresh kernel. The working
directory must be `notebooks/` because the notebook uses relative paths.

The notebook loads `../data/internet_service_churn.csv` and writes:

- `../models/churn_model.pkl`
- `../data/predictions.csv`

Both grid searches may take time. Rebuilding Docker afterwards includes
the regenerated files in the application image.

## Limitations

- The dataset source URL, license and feature units should be confirmed
  before interpreting quantities such as usage or subscription duration.
- Some exploratory analysis uses the full dataset; the test set was not
  completely excluded from exploratory inspection.
- Test performance does not establish future performance or probability calibration.
- The current form uses a shared upper bound of 900. This restricts some
  valid download values in the dataset; field-specific bounds are needed.
- Download and upload inputs are currently required, although the trained
  pipeline can impute missing values in these columns.

## Conclusion

The project demonstrates an end-to-end machine learning workflow for
telecom customer churn prediction: data inspection, preprocessing,
feature engineering, model tuning with cross-validation, evaluation,
model persistence, and integration into an interactive Dash
application.

In the completed experiment, Random Forest achieved an F1-score of
**0.946** and ROC-AUC of **0.981** on the held-out test set.
Contract-related information was especially important for churn
prediction, including whether the remaining-contract value was missing.
