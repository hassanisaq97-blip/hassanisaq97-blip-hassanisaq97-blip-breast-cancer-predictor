# app.py
import joblib, numpy as np
import streamlit as st
import matplotlib.pyplot as plt

SUMMARY = joblib.load("models/summary.joblib")
FEATURES = SUMMARY["features"]
STATS = SUMMARY["feature_stats"]
MODELS = SUMMARY["models"]

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("Breast Cancer Predictor")
st.caption("End-to-end ML • Model comparison • Interactive predictions")

# ----- Tabs -----
tab_overview, tab_perf, tab_importance, tab_predict = st.tabs(
    ["Overview", "Performance", "Feature importance", "Predictor"]
)

# ----- Overview -----
with tab_overview:
    st.subheader("Project overview")
    st.markdown(
        "- Dataset: UCI WDBC\n"
        "- Models: Logistic Regression, RandomForest, SVM (RBF)\n"
        "- Split: stratified hold-out (20%)\n"
        "- Metrics: Accuracy, F1, ROC AUC\n"
        "- Best model (AUC): **{}** ({:.4f})".format(
            SUMMARY["best_model_name"], SUMMARY["best_model_auc"]
        )
    )

# ----- Performance -----
with tab_perf:
    st.subheader("Metrics comparison")
    # Build table
    rows = []
    for name, s in MODELS.items():
        m = s["metrics"]
        rows.append({
            "Model": name,
            "Accuracy": round(m["accuracy"], 4),
            "F1": round(m["f1"], 4),
            "ROC AUC": round(m["roc_auc"], 4),
        })
    st.table(rows)

    st.subheader("ROC curves")
    to_plot = st.multiselect(
        "Choose models to plot",
        options=list(MODELS.keys()),
        default=list(MODELS.keys())
    )
    fig = plt.figure()
    for name in to_plot:
        m = MODELS[name]["metrics"]
        plt.plot(m["fpr"], m["tpr"], label=f"{name} (AUC={m['roc_auc']:.3f})")
    plt.plot([0,1],[0,1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(fig)

# ----- Feature importance -----
with tab_importance:
    st.subheader("Top features")
    model_for_importance = st.selectbox(
        "Model for importance (LR & RF supported)",
        options=["logreg", "random_forest"]
    )
    top = MODELS[model_for_importance]["top_features"]
    if not top:
        st.info("No native importance for this model.")
    else:
        # Bar chart (matplotlib) — sort by value
        names = list(top.keys())
        vals = list(top.values())
        order = np.argsort(vals)[::-1]
        fig2 = plt.figure()
        plt.bar([names[i] for i in order], [vals[i] for i in order])
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Importance (abs coef / impurity)")
        st.pyplot(fig2)

# ----- Predictor -----
with tab_predict:
    st.subheader("Choose model for prediction")
    model_choice = st.selectbox("Model", options=list(MODELS.keys()),
                                index=list(MODELS.keys()).index(SUMMARY["best_model_name"]))

    # Load estimator + scaler (if any)
    est_path = "models/" + MODELS[model_choice]["estimator_file"]
    est = joblib.load(est_path)
    scaler = None
    if MODELS[model_choice]["uses_scaler"]:
        scaler_path = "models/" + MODELS[model_choice]["scaler_file"]
        scaler = joblib.load(scaler_path)

    prefill = st.radio("Prefill", ["Median", "Random", "Zeros"], index=0, horizontal=True)

    st.write("Provide feature values:")
    inputs = []
    for f in FEATURES:
        mn = float(STATS["min"][f]); mx = float(STATS["max"][f]); md = float(STATS["median"][f])
        if prefill == "Median":
            default = md
        elif prefill == "Random":
            default = float(np.random.default_rng().uniform(mn, mx))
        else:
            default = 0.0
        val = st.number_input(f, value=default, min_value=mn, max_value=mx)
        inputs.append(val)

    if st.button("Predict"):
        x = np.array(inputs, dtype=float).reshape(1, -1)
        if scaler is not None:
            x = scaler.transform(x)
        proba = float(est.predict_proba(x)[0, 1])
        pred = int(proba >= 0.5)
        label = "Malignant (1)" if pred == 1 else "Benign (0)"
        st.success(f"Prediction: {label} — Probability malignant: {proba:.2%}")
