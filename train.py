# train.py
# train.py
import os, joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)

DATA_FILE = Path("breast+cancer+wisconsin+diagnostic") / "wdbc.data"
MODELS_DIR = Path("models")
SUMMARY_FILE = MODELS_DIR / "summary.joblib"

COLS = ["id","diagnosis",
"radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean",
"concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
"radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se",
"concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
"radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst",
"concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"]

RANDOM_STATE = 42

def eval_and_curves(est, X_test, y_test, proba_is_decision=False):
    """Return metrics dict + ROC curve arrays."""
    if proba_is_decision:
        # Hvis man brugte decision_function i stedet for predict_proba
        scores = est.decision_function(X_test)
        # Skaler til 0-1 via min-max for AUC/ROC (alternativt brug sigmoid)
        s_min, s_max = scores.min(), scores.max()
        y_score = (scores - s_min) / (s_max - s_min + 1e-8)
    else:
        y_score = est.predict_proba(X_test)[:, 1]

    y_pred = (y_score >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "confusion_matrix": cm,
    }

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Kunne ikke finde datafilen: {DATA_FILE}")

    # --- Load & preprocess ---
    df = pd.read_csv(DATA_FILE, header=None, names=COLS)
    df = df.drop(columns=["id"])
    df["diagnosis"] = df["diagnosis"].map({"M":1, "B":0}).astype(int)

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Feature stats til app inputs
    feature_stats = {
        "min": X.min().to_dict(),
        "max": X.max().to_dict(),
        "median": X.median().to_dict(),
    }

    MODELS_DIR.mkdir(exist_ok=True)

    summaries = {}
    best_model_name = None
    best_auc = -1.0

    # --- 1) Logistic Regression (med scaler) ---
    scaler_lr = StandardScaler()
    X_train_lr = scaler_lr.fit_transform(X_train)
    X_test_lr  = scaler_lr.transform(X_test)

    lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    lr.fit(X_train_lr, y_train)
    m_lr = eval_and_curves(lr, X_test_lr, y_test)

    # LR feature importance (abs(kof))
    lr_importance = pd.Series(np.abs(lr.coef_[0]), index=X.columns).sort_values(ascending=False)
    lr_top = lr_importance.head(10).to_dict()

    joblib.dump(lr, MODELS_DIR / "logreg_est.joblib")
    joblib.dump(scaler_lr, MODELS_DIR / "logreg_scaler.joblib")

    summaries["logreg"] = {
        "metrics": m_lr,
        "top_features": lr_top,
        "uses_scaler": True,
        "estimator_file": "logreg_est.joblib",
        "scaler_file": "logreg_scaler.joblib",
    }
    if m_lr["roc_auc"] > best_auc:
        best_auc = m_lr["roc_auc"]
        best_model_name = "logreg"

    # --- 2) RandomForest (uden scaler) ---
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=2,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    m_rf = eval_and_curves(rf, X_test, y_test)

    # RF feature importance
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    rf_top = rf_importance.head(10).to_dict()

    joblib.dump(rf, MODELS_DIR / "rf_est.joblib")

    summaries["random_forest"] = {
        "metrics": m_rf,
        "top_features": rf_top,
        "uses_scaler": False,
        "estimator_file": "rf_est.joblib",
        "scaler_file": None,
    }
    if m_rf["roc_auc"] > best_auc:
        best_auc = m_rf["roc_auc"]
        best_model_name = "random_forest"

    # --- 3) SVM (RBF) med probability ---
    scaler_svm = StandardScaler()
    X_train_svm = scaler_svm.fit_transform(X_train)
    X_test_svm  = scaler_svm.transform(X_test)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train_svm, y_train)
    m_svm = eval_and_curves(svm, X_test_svm, y_test)

    # SVM har ikke direkte feature importance (springes over)
    joblib.dump(svm, MODELS_DIR / "svm_est.joblib")
    joblib.dump(scaler_svm, MODELS_DIR / "svm_scaler.joblib")

    summaries["svm_rbf"] = {
        "metrics": m_svm,
        "top_features": None,
        "uses_scaler": True,
        "estimator_file": "svm_est.joblib",
        "scaler_file": "svm_scaler.joblib",
    }
    if m_svm["roc_auc"] > best_auc:
        best_auc = m_svm["roc_auc"]
        best_model_name = "svm_rbf"

    # --- Save summary bundle ---
    summary = {
        "features": X.columns.tolist(),
        "feature_stats": feature_stats,
        "models": summaries,
        "best_model_name": best_model_name,
        "best_model_auc": best_auc,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(summary, SUMMARY_FILE)

    # Konsolfeedback
    print("=== Hold-out metrics ===")
    for name, s in summaries.items():
        print(f"[{name}] acc={s['metrics']['accuracy']:.4f}  f1={s['metrics']['f1']:.4f}  auc={s['metrics']['roc_auc']:.4f}")
    print(f"Saved summary → {SUMMARY_FILE}")
    print("Estimators saved in /models/*.joblib")

if __name__ == "__main__":
    main()
