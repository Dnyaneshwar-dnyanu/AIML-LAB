import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import Bunch


def load_dataset():
    """
    Try to load the California Housing dataset; if unavailable (no internet),
    fall back to the Diabetes dataset.
    """
    try:
        from sklearn.datasets import fetch_california_housing
        ds = fetch_california_housing(as_frame=True)
        name = "California Housing"
        X = ds.frame.drop(columns=[ds.target_names[0]])
        y = ds.frame[ds.target_names[0]]
        feature_names = list(X.columns)
        return Bunch(data=X, target=y, feature_names=feature_names, name=name)
    except Exception:
        from sklearn.datasets import load_diabetes
        ds = load_diabetes(as_frame=True)
        name = "Diabetes (fallback)"
        X = ds.frame.drop(columns=["target"])
        y = ds.frame["target"]
        feature_names = list(X.columns)
        return Bunch(data=X, target=y, feature_names=feature_names, name=name)


def quick_eda(X, y, name):
    """
    Print dataset summary and save a histogram of the target variable.
    """
    print(f"Dataset: {name} | samples={len(X)} | features={len(X.columns)}")
    print("Head:\n", X.head(3))
    print("Target stats: mean={:.3f} std={:.3f}".format(y.mean(), y.std()))

    # Tiny histogram of target
    plt.figure()
    y.hist(bins=30)
    plt.title(f"{name} target distribution")
    plt.xlabel("y")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("target_hist.png")


def build_and_eval(X, y, feature_names):
    """
    Build preprocessing + Linear Regression pipeline, evaluate, and save plots.
    """
    num_features = list(feature_names)
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(with_mean=True, with_std=True), num_features)],
        remainder="drop"
    )

    pipe = Pipeline([
        ("pre", pre),
        ("lr", LinearRegression())
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)

    rmse = np.sqrt(mean_squared_error(yte, preds))
    mae = mean_absolute_error(yte, preds)
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # Pred vs True plot
    plt.figure()
    plt.scatter(yte, preds, s=10)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Linear Regression: True vs Predicted")
    lims = [min(yte.min(), preds.min()), max(yte.max(), preds.max())]
    plt.plot(lims, lims)
    plt.tight_layout()
    plt.savefig("true_vs_pred.png")

    # Model card metadata
    card = {
        "model": "LinearRegression",
        "dataset": "California Housing (fallback to Diabetes if offline)",
        "task": "Tabular regression",
        "preprocessing": "StandardScaler on all numeric features",
        "target": "MedianHouseValue (or Diabetes target)",
        "metrics": {"RMSE": float(rmse), "MAE": float(mae)},
        "intended_use": "Intro ML coursework; not for real-estate decisions",
        "limitations": [
            "Linear model; no interaction/nonlinearity modeling",
            "No feature engineering; sensitive to scale/outliers"
        ],
        "owner": "Student",
    }

    with open("model_card.json", "w") as f:
        json.dump(card, f, indent=2)

    print("Saved: target_hist.png, true_vs_pred.png, model_card.json")

    return rmse, mae


if __name__ == "__main__":
    ds = load_dataset()
    quick_eda(ds.data, ds.target, ds.name)
    build_and_eval(ds.data, ds.target, ds.feature_names)
