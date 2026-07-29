import json
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"


def generate_ai_summary(prediction, probability, explanation_table):
    """
    Generate a short explanation using the local Ollama model.
    """

    feature_lines = []

    for row in explanation_table:
        feature_lines.append(
            f"- {row['Feature']}: SHAP value {row['SHAP value']:.4f}, "
            f"{row['Effect']}"
        )

    features_text = "\n".join(feature_lines)

    prompt = f"""
You are explaining the output of a machine-learning demonstration.

Prediction: {prediction}
Probability of malignancy: {probability:.2%}

Most influential features:
{features_text}

Write a clear explanation in 3 to 4 short sentences.
Explain which features pushed the prediction toward malignant or benign.
Do not give medical advice.
State that the result is from a demonstration model and is not a diagnosis.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["response"].strip()

    except urllib.error.URLError:
        return (
            "The local AI model could not be reached. "
            "Make sure Ollama is running."
        )