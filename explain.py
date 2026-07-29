import pandas as pd
import shap


def explain_prediction(estimator, input_data, feature_names, background_data):
    """
    Generate a SHAP explanation for a single prediction.
    """

    x = pd.DataFrame(input_data, columns=feature_names)

    background = pd.DataFrame(
        background_data,
        columns=feature_names,
    )

    explainer = shap.Explainer(estimator, background)

    shap_values = explainer(x)

    return shap_values