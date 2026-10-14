# -*- coding: utf-8 -*-
"""
================================================================================
AIoT HW2 - Red Wine Quality Prediction with Linear Regression
================================================================================
Dataset : Red Wine Quality (UCI / Kaggle)
Source  : https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
Target  : quality (0-10 quality score, treated as a continuous target)
Features: 11 physicochemical measurements

This script follows the CRISP-DM methodology:
    1. Business Understanding
    2. Data Understanding
    3. Data Preparation
    4. Modeling            (Simple / Multiple Linear Regression + Ridge + Lasso)
    5. Evaluation          (MAE / MSE / RMSE / R2 / Adjusted R2 + assumptions)
    6. Deployment          (predict a new sample + save artifacts)

It performs Feature Selection (SelectKBest, RFE, Lasso) and Model Evaluation,
and produces a prediction plot that includes a 95% prediction interval and a
95% confidence interval for the regression line.

Run:
    python 7114056032_hw2.py
================================================================================
"""

import os
import json
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

try:
    import requests  # for web data acquisition (optional dependency)
except ImportError:
    requests = None

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import statsmodels.api as sm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "winequality-red.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# Export directory for the web dashboard (Vercel-ready static site).
WEB_DATA_DIR = os.path.join(BASE_DIR, "web", "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEB_DATA_DIR, exist_ok=True)

# Raw CSV on the web (same columns as the local copy). Mirrors the 50-Startups
# approach of fetching data over HTTP with the `requests` module.
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "quality"

# Collected throughout the run and dumped to web/data/results.json at the end.
WEB_RESULTS = {}


def banner(title):
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


# --------------------------------------------------------------------------- #
# 1. BUSINESS UNDERSTANDING
# --------------------------------------------------------------------------- #
def business_understanding():
    banner("1. BUSINESS UNDERSTANDING")
    print(
        "Goal: Predict the sensory quality score of red wine from 11 physico-\n"
        "chemical lab measurements (acidity, sulphates, alcohol, etc.).\n"
        "A reliable model lets producers estimate quality from objective tests\n"
        "instead of relying only on expensive, subjective human tasting panels.\n"
        "This is a supervised regression problem; the target 'quality' is an\n"
        "integer score (3-8 in this data) that we model as continuous."
    )


# --------------------------------------------------------------------------- #
# Data acquisition - fetch from the web (requests), fall back to local file.
# This mirrors the 50-Startups workflow: GET the CSV, verify HTTP 200,
# read into a DataFrame, and fail gracefully if anything goes wrong.
# --------------------------------------------------------------------------- #
def _read_csv_text(text):
    """Robustly parse CSV text whether it uses ',' or ';' as the separator."""
    header = text.splitlines()[0] if text else ""
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(StringIO(text), sep=sep)
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def load_data():
    """Step 1 - Data Acquisition.

    Try to download the dataset from DATA_URL using `requests` and verify the
    HTTP status code is 200. On any failure (no requests module, network error,
    non-200 status) fall back to the bundled local CSV so the script always
    runs.
    """
    source = None
    df = None

    if requests is not None:
        try:
            print(f"Fetching data from: {DATA_URL}")
            resp = requests.get(DATA_URL, timeout=15)
            if resp.status_code == 200:
                df = _read_csv_text(resp.text)
                source = "web"
                print(f"HTTP 200 OK - downloaded {len(df)} rows from the web.")
            else:
                print(f"Failed to retrieve data, status code: {resp.status_code}")
        except Exception as exc:  # network / parse errors -> graceful fallback
            print(f"Web fetch failed ({exc!r}); falling back to local file.")
    else:
        print("`requests` not installed; using local file.")

    if df is None:
        with open(DATA_PATH, "r", encoding="utf-8") as fh:
            df = _read_csv_text(fh.read())
        source = "local"
        print(f"Loaded {len(df)} rows from local file: {DATA_PATH}")

    WEB_RESULTS["data_source"] = source
    WEB_RESULTS["data_url"] = DATA_URL
    return df


# --------------------------------------------------------------------------- #
# 2. DATA UNDERSTANDING
# --------------------------------------------------------------------------- #
def data_understanding():
    banner("2. DATA UNDERSTANDING")

    df = load_data()

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Features ({df.shape[1] - 1}): {[c for c in df.columns if c != TARGET]}")
    print(f"Target: {TARGET}")
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())
    print("\nStatistical summary:")
    print(df.describe().round(3).to_string())
    print("\nTarget distribution:")
    print(df[TARGET].value_counts().sort_index().to_string())

    # --- Figure: target distribution ---
    plt.figure(figsize=(7, 4.5))
    sns.countplot(x=TARGET, data=df, color="#8e3b46")
    plt.title("Target Distribution: Red Wine Quality Scores")
    plt.xlabel("Quality score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "01_target_distribution.png"), dpi=150)
    plt.close()

    # --- Figure: correlation heatmap ---
    plt.figure(figsize=(11, 9))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap of Features and Target")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "02_correlation_heatmap.png"), dpi=150)
    plt.close()

    target_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    print("\nTop correlations with target (|r|):")
    print(target_corr.round(3).to_string())
    print("\nSaved: 01_target_distribution.png, 02_correlation_heatmap.png")

    # --- collect for the web dashboard ---
    dist = df[TARGET].value_counts().sort_index()
    WEB_RESULTS["n_rows"] = int(df.shape[0])
    WEB_RESULTS["n_features"] = int(df.shape[1] - 1)
    WEB_RESULTS["feature_names"] = [c for c in df.columns if c != TARGET]
    WEB_RESULTS["target"] = TARGET
    WEB_RESULTS["missing_total"] = int(df.isnull().sum().sum())
    WEB_RESULTS["target_distribution"] = {
        "labels": [str(int(s)) for s in dist.index.tolist()],
        "counts": [int(c) for c in dist.values.tolist()],
    }
    WEB_RESULTS["target_correlation"] = {
        "features": target_corr.index.tolist(),
        "abs_r": [round(float(v), 4) for v in target_corr.values.tolist()],
        "signed_r": [round(float(corr[TARGET][f]), 4) for f in target_corr.index],
    }
    return df


# --------------------------------------------------------------------------- #
# 3. DATA PREPARATION
# --------------------------------------------------------------------------- #
def data_preparation(df):
    banner("3. DATA PREPARATION")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows -> {len(df)} rows remain.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train/Test split: {len(X_train)} / {len(X_test)} "
          f"({int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%)")

    # Standardize features (fit on train only to avoid data leakage).
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                             columns=X.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test),
                            columns=X.columns, index=X_test.index)
    print("Standardized features with StandardScaler (fit on training set only).")

    return X_train_s, X_test_s, y_train, y_test, X.columns.tolist()


# --------------------------------------------------------------------------- #
# 3b. FEATURE SELECTION
# --------------------------------------------------------------------------- #
def feature_selection(X_train, y_train, feature_names, k=8):
    banner("3b. FEATURE SELECTION")

    # Method 1: SelectKBest (univariate F-test)
    skb = SelectKBest(score_func=f_regression, k=k).fit(X_train, y_train)
    skb_scores = pd.Series(skb.scores_, index=feature_names).sort_values(ascending=False)
    skb_selected = skb_scores.head(k).index.tolist()
    print(f"[SelectKBest / f_regression] top {k}:")
    print(skb_scores.round(2).to_string())

    # Method 2: RFE with Linear Regression
    rfe = RFE(LinearRegression(), n_features_to_select=k).fit(X_train, y_train)
    rfe_selected = [f for f, keep in zip(feature_names, rfe.support_) if keep]
    print(f"\n[RFE / LinearRegression] selected {k}: {rfe_selected}")

    # Method 3: Lasso (L1) - features with non-zero coefficients
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000).fit(X_train, y_train)
    lasso_coef = pd.Series(lasso.coef_, index=feature_names)
    lasso_selected = lasso_coef[lasso_coef.abs() > 1e-6].index.tolist()
    print(f"\n[LassoCV] alpha={lasso.alpha_:.4f}, non-zero features "
          f"({len(lasso_selected)}): {lasso_selected}")

    # Consensus: features chosen by at least 2 of the 3 methods
    votes = pd.Series(0, index=feature_names)
    for sel in (skb_selected, rfe_selected, lasso_selected):
        for f in sel:
            votes[f] += 1
    consensus = votes[votes >= 2].index.tolist()
    print(f"\nConsensus features (chosen by >=2 methods): {consensus}")

    # --- Figure: SelectKBest scores ---
    plt.figure(figsize=(8, 5))
    skb_scores.sort_values().plot(kind="barh", color="#4c72b0")
    plt.title("Feature Importance (SelectKBest F-score)")
    plt.xlabel("F-score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "03_feature_selection.png"), dpi=150)
    plt.close()
    print("Saved: 03_feature_selection.png")

    # --- collect for the web dashboard ---
    WEB_RESULTS["feature_selection"] = {
        "skb_features": skb_scores.index.tolist(),
        "skb_scores": [round(float(v), 2) for v in skb_scores.values.tolist()],
        "rfe_selected": rfe_selected,
        "lasso_selected": lasso_selected,
        "lasso_alpha": round(float(lasso.alpha_), 4),
        "consensus": consensus,
    }

    return consensus if len(consensus) >= 3 else skb_selected


# --------------------------------------------------------------------------- #
# 4. MODELING
# --------------------------------------------------------------------------- #
def modeling(X_train, X_test, y_train, y_test, selected):
    banner("4. MODELING")

    results = {}

    def fit_eval(name, model, Xtr, Xte):
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        results[name] = {
            "model": model,
            "pred": pred,
            "n_features": Xtr.shape[1],
            "mae": mean_absolute_error(y_test, pred),
            "mse": mean_squared_error(y_test, pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": r2_score(y_test, pred),
        }
        # 5-fold CV R2 on training data
        cv = cross_val_score(model, Xtr, y_train, cv=KFold(5, shuffle=True,
                             random_state=RANDOM_STATE), scoring="r2")
        results[name]["cv_r2"] = cv.mean()

    # (a) Simple linear regression - single best predictor (alcohol)
    best_feat = "alcohol"
    fit_eval(f"Simple LR ({best_feat})", LinearRegression(),
             X_train[[best_feat]], X_test[[best_feat]])

    # (b) Multiple linear regression - all features
    fit_eval("Multiple LR (all)", LinearRegression(), X_train, X_test)

    # (c) Multiple linear regression - selected features
    fit_eval("Multiple LR (selected)", LinearRegression(),
             X_train[selected], X_test[selected])

    # (d) Ridge (L2 regularized)
    fit_eval("Ridge", RidgeCV(alphas=np.logspace(-3, 3, 50)), X_train, X_test)

    # (e) Lasso (L1 regularized)
    fit_eval("Lasso", LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000),
             X_train, X_test)

    print(f"Simple LR uses 1 feature: '{best_feat}'")
    print(f"Selected-feature MLR uses {len(selected)} features: {selected}")
    return results


# --------------------------------------------------------------------------- #
# 5. EVALUATION
# --------------------------------------------------------------------------- #
def adjusted_r2(r2, n, p):
    denom = (n - p - 1)
    return 1 - (1 - r2) * (n - 1) / denom if denom > 0 else float("nan")


def evaluation(results, X_test, y_test):
    banner("5. EVALUATION")

    n_test = len(y_test)
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "n_feat": r["n_features"],
            "MAE": r["mae"],
            "MSE": r["mse"],
            "RMSE": r["rmse"],
            "R2": r["r2"],
            "Adj_R2": adjusted_r2(r["r2"], n_test, r["n_features"]),
            "CV_R2": r["cv_r2"],
        })
    metrics = pd.DataFrame(rows).set_index("Model")
    print(metrics.round(4).to_string())
    metrics.round(4).to_csv(os.path.join(RESULTS_DIR, "metrics.csv"))
    print("\nSaved: metrics.csv")

    # --- Figure: model comparison (RMSE & R2) ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    metrics["RMSE"].plot(kind="bar", ax=ax[0], color="#c44e52")
    ax[0].set_title("Test RMSE by Model (lower is better)")
    ax[0].set_ylabel("RMSE")
    ax[0].tick_params(axis="x", rotation=30)
    metrics["R2"].plot(kind="bar", ax=ax[1], color="#55a868")
    ax[1].set_title("Test R2 by Model (higher is better)")
    ax[1].set_ylabel("R2")
    ax[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "04_model_comparison.png"), dpi=150)
    plt.close()
    print("Saved: 04_model_comparison.png")

    # Pick best model by test R2 among the multiple-regression variants.
    best_name = metrics["R2"].idxmax()
    print(f"\nBest model by test R2: '{best_name}' "
          f"(R2={metrics.loc[best_name, 'R2']:.4f}, "
          f"RMSE={metrics.loc[best_name, 'RMSE']:.4f})")

    # --- collect metrics table for the web dashboard ---
    WEB_RESULTS["metrics"] = [
        {
            "model": idx,
            "n_feat": int(row["n_feat"]),
            "mae": round(float(row["MAE"]), 4),
            "mse": round(float(row["MSE"]), 4),
            "rmse": round(float(row["RMSE"]), 4),
            "r2": round(float(row["R2"]), 4),
            "adj_r2": round(float(row["Adj_R2"]), 4),
            "cv_r2": round(float(row["CV_R2"]), 4),
        }
        for idx, row in metrics.iterrows()
    ]
    WEB_RESULTS["best_model"] = best_name

    # --- actual vs predicted scatter (best model) for the website ---
    best = results[best_name]
    WEB_RESULTS["actual_vs_predicted"] = {
        "actual": [float(v) for v in y_test.values.tolist()],
        "predicted": [round(float(v), 3) for v in best["pred"].tolist()],
    }

    # --- Residual diagnostics for the best model (regression assumptions) ---
    residuals = y_test.values - best["pred"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(best["pred"], residuals, alpha=0.5, color="#4c72b0", edgecolor="k", linewidth=0.3)
    ax[0].axhline(0, color="red", ls="--")
    ax[0].set_title("Residuals vs Fitted (linearity / homoscedasticity)")
    ax[0].set_xlabel("Fitted values")
    ax[0].set_ylabel("Residuals")
    stats.probplot(residuals, dist="norm", plot=ax[1])
    ax[1].set_title("Normal Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "05_residual_diagnostics.png"), dpi=150)
    plt.close()
    print("Saved: 05_residual_diagnostics.png")

    return best_name, metrics


# --------------------------------------------------------------------------- #
# 5b. PREDICTION PLOT WITH CONFIDENCE / PREDICTION INTERVALS
# --------------------------------------------------------------------------- #
def prediction_interval_plot(df):
    """
    Fit a simple OLS of quality ~ alcohol using statsmodels so we can draw both
    a 95% confidence interval (for the mean line) and a 95% prediction interval
    (for individual observations). This directly satisfies the assignment's
    requirement for a prediction plot WITH intervals.
    """
    banner("5c. PREDICTION PLOT WITH CONFIDENCE & PREDICTION INTERVALS")

    x = df["alcohol"].values
    y = df[TARGET].values
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    print(ols.summary().tables[1].as_text())

    x_grid = np.linspace(x.min(), x.max(), 200)
    Xg = sm.add_constant(x_grid)
    pred = ols.get_prediction(Xg)
    sf = pred.summary_frame(alpha=0.05)  # 95%

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, alpha=0.25, color="#8e3b46", label="Observed", edgecolor="none")
    plt.plot(x_grid, sf["mean"], color="navy", lw=2, label="OLS fit")
    plt.fill_between(x_grid, sf["mean_ci_lower"], sf["mean_ci_upper"],
                     color="navy", alpha=0.30, label="95% confidence interval (mean)")
    plt.fill_between(x_grid, sf["obs_ci_lower"], sf["obs_ci_upper"],
                     color="orange", alpha=0.20, label="95% prediction interval")
    plt.title("Red Wine Quality vs Alcohol\nOLS fit with 95% Confidence & Prediction Intervals")
    plt.xlabel("Alcohol (% vol)")
    plt.ylabel("Quality score")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "06_prediction_intervals.png"), dpi=150)
    plt.close()
    print("Saved: 06_prediction_intervals.png")

    # --- collect for the web dashboard ---
    WEB_RESULTS["ols_alcohol"] = {
        "intercept": round(float(ols.params[0]), 4),
        "slope": round(float(ols.params[1]), 4),
        "r2": round(float(ols.rsquared), 4),
    }
    WEB_RESULTS["prediction_interval"] = {
        "x": [round(float(v), 3) for v in x_grid.tolist()],
        "mean": [round(float(v), 4) for v in sf["mean"].tolist()],
        "ci_lower": [round(float(v), 4) for v in sf["mean_ci_lower"].tolist()],
        "ci_upper": [round(float(v), 4) for v in sf["mean_ci_upper"].tolist()],
        "pi_lower": [round(float(v), 4) for v in sf["obs_ci_lower"].tolist()],
        "pi_upper": [round(float(v), 4) for v in sf["obs_ci_upper"].tolist()],
    }
    # downsample the raw scatter so the JSON stays small
    step = max(1, len(x) // 400)
    WEB_RESULTS["scatter_alcohol"] = {
        "x": [round(float(v), 3) for v in x[::step].tolist()],
        "y": [float(v) for v in y[::step].tolist()],
    }


def actual_vs_predicted_plot(results, best_name, y_test):
    best = results[best_name]
    pred = best["pred"]
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, pred, alpha=0.5, color="#55a868", edgecolor="k", linewidth=0.3)
    lims = [min(y_test.min(), pred.min()) - 0.5, max(y_test.max(), pred.max()) + 0.5]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlim(lims); plt.ylim(lims)
    plt.title(f"Actual vs Predicted Quality\n{best_name}")
    plt.xlabel("Actual quality")
    plt.ylabel("Predicted quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "07_actual_vs_predicted.png"), dpi=150)
    plt.close()
    print("Saved: 07_actual_vs_predicted.png")


# --------------------------------------------------------------------------- #
# 6. DEPLOYMENT
# --------------------------------------------------------------------------- #
def deployment(results, best_name, X_test, y_test, feature_names):
    banner("6. DEPLOYMENT")

    best = results[best_name]
    model = best["model"]

    # Determine the feature subset this model was trained on.
    n = best["n_features"]
    if n == 1:
        cols = ["alcohol"]
    elif n == len(feature_names):
        cols = feature_names
    else:
        # selected-feature model: infer from coefficient length via X_test slice
        cols = list(X_test.columns)[:n] if n != X_test.shape[1] else list(X_test.columns)

    # Report coefficients for interpretability.
    if hasattr(model, "coef_") and np.ndim(model.coef_) == 1 and len(model.coef_) == len(_match_cols(best, X_test, feature_names)):
        used = _match_cols(best, X_test, feature_names)
        coef = pd.Series(model.coef_, index=used).sort_values(key=np.abs, ascending=False)
        print("Standardized coefficients (impact on quality):")
        print(coef.round(4).to_string())
        print(f"Intercept: {model.intercept_:.4f}")
        WEB_RESULTS["coefficients"] = {
            "features": coef.index.tolist(),
            "values": [round(float(v), 4) for v in coef.values.tolist()],
            "intercept": round(float(model.intercept_), 4),
        }

    # Demonstrate prediction on the first 5 test samples.
    used = _match_cols(best, X_test, feature_names)
    sample = X_test[used].head(5)
    preds = model.predict(sample)
    demo = pd.DataFrame({"actual": y_test.head(5).values,
                         "predicted": np.round(preds, 2)})
    print("\nDeployment demo - predicting 5 unseen samples:")
    print(demo.to_string(index=False))
    print("\nIn production the fitted model + StandardScaler would be serialized "
          "(e.g. joblib) and wrapped in an API/batch job to score new wines.")
    WEB_RESULTS["demo_predictions"] = [
        {"actual": float(a), "predicted": round(float(p), 2)}
        for a, p in zip(demo["actual"].tolist(), demo["predicted"].tolist())
    ]


# --------------------------------------------------------------------------- #
# Export all collected results as JSON for the web dashboard.
# --------------------------------------------------------------------------- #
def export_web_results():
    banner("EXPORT: web/data/results.json")
    out_path = os.path.join(WEB_DATA_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(WEB_RESULTS, fh, ensure_ascii=False, indent=2)
    print(f"Saved dashboard data: {out_path}")

    # Also copy the generated figures into the web folder so the static site
    # can display them without referencing files outside its root.
    import shutil
    web_img_dir = os.path.join(BASE_DIR, "web", "images")
    os.makedirs(web_img_dir, exist_ok=True)
    for fname in os.listdir(RESULTS_DIR):
        if fname.lower().endswith(".png"):
            shutil.copy2(os.path.join(RESULTS_DIR, fname),
                         os.path.join(web_img_dir, fname))
    print(f"Copied result figures to: {web_img_dir}")


def _match_cols(best, X_test, feature_names):
    """Resolve which columns a model used based on its feature count."""
    n = best["n_features"]
    if n == 1:
        return ["alcohol"]
    if n == len(feature_names):
        return feature_names
    # selected feature model - stored separately
    return SELECTED_FEATURES


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
SELECTED_FEATURES = []

def main():
    global SELECTED_FEATURES
    business_understanding()
    df = data_understanding()
    X_train, X_test, y_train, y_test, feature_names = data_preparation(df)
    SELECTED_FEATURES = feature_selection(X_train, y_train, feature_names, k=8)
    results = modeling(X_train, X_test, y_train, y_test, SELECTED_FEATURES)
    best_name, metrics = evaluation(results, X_test, y_test)
    prediction_interval_plot(df)
    actual_vs_predicted_plot(results, best_name, y_test)
    deployment(results, best_name, X_test, y_test, feature_names)
    export_web_results()

    banner("DONE")
    print(f"All figures and metrics saved to: {RESULTS_DIR}")
    print(f"Web dashboard data saved to: {WEB_DATA_DIR}")


if __name__ == "__main__":
    main()
