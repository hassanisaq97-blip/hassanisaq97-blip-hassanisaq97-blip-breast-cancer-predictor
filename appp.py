from pathlib import Path

corrected_code = '''# app.py
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from audit_log import log_prediction
from explain import explain_prediction
from llm import generate_ai_summary


# -----------------------------
# Configuration and data loading
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_summary():
    return joblib.load("models/summary.joblib")


@st.cache_resource
def load_estimator(model_info):
    estimator_path = Path("models") / model_info["estimator_file"]
    estimator = joblib.load(estimator_path)

    scaler = None
    if model_info["uses_scaler"]:
        scaler_path = Path("models") / model_info["scaler_file"]
        scaler = joblib.load(scaler_path)

    return estimator, scaler


SUMMARY = load_summary()
FEATURES = SUMMARY["features"]
STATS = SUMMARY["feature_stats"]
MODELS = SUMMARY["models"]


# -----------------------------
# Page header
# -----------------------------
st.title("Breast Cancer Predictor")
st.caption(
    "End-to-end machine learning • Model comparison • "
    "Explainable predictions • Local AI summary • Audit logging"
)


# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_perf, tab_importance, tab_predict = st.tabs(
    ["Overview", "Performance", "Feature importance", "Predictor"]
)


# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    st.subheader("Project overview")
    st.markdown(
        "- Dataset: UCI Wisconsin Diagnostic Breast Cancer Dataset\\n"
        "- Models: Logistic Regression, Random Forest and SVM (RBF)\\n"
        "- Split: Stratified hold-out test set (20%)\\n"
        "- Metrics: Accuracy, F1 score and ROC AUC\\n"
        "- Explainability: SHAP feature effects and waterfall plot\\n"
        "- Local AI: Llama 3.2 through Ollama\\n"
        "- Audit trail: Predictions stored in SQLite\\n"
        "- Best model by ROC AUC: **{}** ({:.4f})".format(
            SUMMARY["best_model_name"],
            SUMMARY["best_model_auc"],
        )
    )

    st.warning(
        "Educational project only. The application is not intended for "
        "clinical diagnosis or medical decision-making."
    )


# -----------------------------
# Performance
# -----------------------------
with tab_perf:
    st.subheader("Metrics comparison")

    metric_rows = []
    for model_name, model_info in MODELS.items():
        metrics = model_info["metrics"]
        metric_rows.append(
            {
                "Model": model_name,
                "Accuracy": round(metrics["accuracy"], 4),
                "F1": round(metrics["f1"], 4),
                "ROC AUC": round(metrics["roc_auc"], 4),
            }
        )

    st.dataframe(
        pd.DataFrame(metric_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("ROC curves")

    selected_models = st.multiselect(
        "Choose models to plot",
        options=list(MODELS.keys()),
        default=list(MODELS.keys()),
    )

    roc_figure, roc_axis = plt.subplots(figsize=(9, 6))

    for model_name in selected_models:
        metrics = MODELS[model_name]["metrics"]
        roc_axis.plot(
            metrics["fpr"],
            metrics["tpr"],
            label=f"{model_name} (AUC={metrics['roc_auc']:.3f})",
        )

    roc_axis.plot([0, 1], [0, 1], "--", label="Random classifier")
    roc_axis.set_xlabel("False positive rate")
    roc_axis.set_ylabel("True positive rate")
    roc_axis.legend()
    roc_axis.grid(alpha=0.2)

    st.pyplot(roc_figure)
    plt.close(roc_figure)


# -----------------------------
# Feature importance
# -----------------------------
with tab_importance:
    st.subheader("Top features")

    model_for_importance = st.selectbox(
        "Model for feature importance",
        options=["logreg", "random_forest"],
    )

    top_features = MODELS[model_for_importance]["top_features"]

    if not top_features:
        st.info("Native feature importance is not available for this model.")
    else:
        names = list(top_features.keys())
        values = list(top_features.values())
        order = np.argsort(values)[::-1]

        importance_figure, importance_axis = plt.subplots(figsize=(11, 6))
        importance_axis.bar(
            [names[index] for index in order],
            [values[index] for index in order],
        )
        importance_axis.tick_params(axis="x", rotation=60)
        importance_axis.set_ylabel("Importance (absolute coefficient or impurity)")
        importance_axis.grid(axis="y", alpha=0.2)

        st.pyplot(importance_figure)
        plt.close(importance_figure)


# -----------------------------
# Predictor
# -----------------------------
with tab_predict:
    st.subheader("Create a prediction")

    model_choice = st.selectbox(
        "Model",
        options=list(MODELS.keys()),
        index=list(MODELS.keys()).index(SUMMARY["best_model_name"]),
    )

    estimator, scaler = load_estimator(MODELS[model_choice])

    prefill = st.radio(
        "Prefill",
        ["Median", "Random"],
        index=0,
        horizontal=True,
    )

    generate_local_summary = st.checkbox(
        "Generate local AI summary with Llama 3.2",
        value=False,
        help=(
            "Enable this only when Ollama is running locally. "
            "The prediction and SHAP explanation work without it."
        ),
    )

    st.write("Provide feature values:")

    inputs = []
    random_generator = np.random.default_rng()

    for feature in FEATURES:
        minimum = float(STATS["min"][feature])
        maximum = float(STATS["max"][feature])
        median = float(STATS["median"][feature])

        if prefill == "Median":
            default_value = median
        else:
            default_value = float(
                random_generator.uniform(minimum, maximum)
            )

        value = st.number_input(
            feature,
            min_value=minimum,
            max_value=maximum,
            value=default_value,
            key=f"input_{feature}",
        )
        inputs.append(value)

    if st.button("Predict", type="primary", use_container_width=True):
        raw_input = np.asarray(inputs, dtype=float).reshape(1, -1)
        model_input = raw_input.copy()

        if scaler is not None:
            model_input = scaler.transform(model_input)

        malignancy_probability = float(
            estimator.predict_proba(model_input)[0, 1]
        )
        predicted_class = int(malignancy_probability >= 0.5)
        prediction_label = (
            "Malignant (1)"
            if predicted_class == 1
            else "Benign (0)"
        )

        prediction_column, probability_column = st.columns(2)

        with prediction_column:
            st.metric("Prediction", prediction_label)

        with probability_column:
            st.metric(
                "Probability of malignancy",
                f"{malignancy_probability:.2%}",
            )

        if predicted_class == 1:
            st.warning(
                "The model found patterns associated with malignancy."
            )
        else:
            st.success(
                "The model found patterns associated with a benign result."
            )

        st.subheader("SHAP explanation")

        explanation_table = []
        selected_shap = None

        try:
            background = np.asarray(
                [
                    [
                        float(STATS["median"][feature])
                        for feature in FEATURES
                    ]
                ],
                dtype=float,
            )

            if scaler is not None:
                background = scaler.transform(background)

            shap_values = explain_prediction(
                estimator,
                model_input,
                FEATURES,
                background,
            )

            shap_value_array = np.asarray(shap_values.values)

            if shap_value_array.ndim == 3:
                selected_shap = shap_values[0, :, 1]
            else:
                selected_shap = shap_values[0]

            feature_effects = np.asarray(
                selected_shap.values,
                dtype=float,
            ).reshape(-1)

            explanation_table = [
                {
                    "Feature": feature,
                    "SHAP value": float(effect),
                    "Effect": (
                        "Pushes toward malignant"
                        if effect > 0
                        else "Pushes toward benign"
                    ),
                }
                for feature, effect in zip(FEATURES, feature_effects)
            ]

            explanation_table = sorted(
                explanation_table,
                key=lambda row: abs(row["SHAP value"]),
                reverse=True,
            )[:5]

            st.write("Top five features influencing this prediction:")

            st.dataframe(
                pd.DataFrame(explanation_table),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("SHAP feature impact")

            plot_values = (
                pd.DataFrame(explanation_table)
                .set_index("Feature")["SHAP value"]
            )
            st.bar_chart(plot_values)

            st.subheader("SHAP waterfall plot")

            waterfall_figure = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(
                selected_shap,
                max_display=10,
                show=False,
            )
            st.pyplot(waterfall_figure)
            plt.close(waterfall_figure)

        except Exception as error:
            st.error(
                "The prediction was created, but the SHAP explanation "
                f"could not be generated: {error}"
            )

        ai_summary = "Not generated."

        if generate_local_summary:
            st.subheader("AI summary")

            try:
                with st.spinner(
                    "Generating explanation with Llama 3.2..."
                ):
                    ai_summary = generate_ai_summary(
                        prediction_label,
                        malignancy_probability,
                        explanation_table,
                    )

                st.write(ai_summary)

            except Exception as error:
                ai_summary = f"AI summary failed: {error}"
                st.error(
                    "The prediction was created, but the local AI summary "
                    f"could not be generated: {error}"
                )

        try:
            log_prediction(
                model_name=model_choice,
                prediction=prediction_label,
                probability=malignancy_probability,
                input_features=dict(zip(FEATURES, inputs)),
                explanation_table=explanation_table,
                ai_summary=ai_summary,
            )

            st.caption("Prediction saved to the SQLite audit log.")

        except Exception as error:
            st.warning(
                "The prediction was created, but it could not be saved "
                f"to the audit log: {error}"
            )
'''

output_path = Path("/mnt/data/app_fixed.py")
output_path.write_text(corrected_code, encoding="utf-8")

print(f"Created: {output_path}")
print(f"Lines: {len(corrected_code.splitlines())}")
