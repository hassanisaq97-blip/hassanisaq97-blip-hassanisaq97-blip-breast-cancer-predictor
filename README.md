# 🩺 Explainable Breast Cancer Prediction with Machine Learning & Local LLM

An interactive machine learning application for breast cancer prediction built with **Python** and **Streamlit**. The project combines classical machine learning models with explainable AI (SHAP) and a locally hosted large language model (Llama 3.2 via Ollama) to generate human-readable explanations for each prediction.

---

## Screenshots


### Prediction

![Prediction](screenshots/prediction.png)

### SHAP Explanation

![SHAP Explanation](screenshots/shap.png)

### AI Summary

![AI Summary](screenshots/ai_summary.png)

### Performance og ROC curves
![Performance](assets/performance.png)

### Feature importance
![Feature Importance](assets/importance.png)

### Predictor
![Predictor](assets/predictor.png)

## Resultater (hold-out 20%)

| Model          | Accuracy | F1    | ROC AUC |
|----------------|---------:|------:|--------:|
| LogisticReg    | 0.9649   | 0.9512| 0.9960  |
| RandomForest   | 0.9649   | 0.9512| 0.9942  |
| SVM (RBF)      | 0.9649   | 0.9524| 0.9947  |

Resultaterne stammer fra `python train.py` og kan genskabes.

## Projektstruktur

breast-cancer-predictor/
├── app.py # Streamlit-app (tabs: overview, performance, importance, predictor)
├── train.py # Træner 3 modeller, gemmer estimators + summary.joblib
├── train.py.ipynb # Notebook med analyse/EDA og træning
├── models/
│ ├── summary.joblib # features + metrics + ROC-kurver + filstier til estimators
│ ├── logreg_est.joblib # Logistic Regression estimator
│ ├── logreg_scaler.joblib # Scaler til LR
│ ├── rf_est.joblib # RandomForest estimator
│ ├── svm_est.joblib # SVM (RBF) estimator
│ └── svm_scaler.joblib # Scaler til SVM
├── assets/ # Screenshots til README
│ ├── performance.png
│ ├── importance.png
│ └── predictor.png
├── requirements.txt # Python afhængigheder
└── (data ikke committet – se nedenfor)


## Sådan kører du projektet

### 1. Klon repo og gå ind i mappen
```bash
git clone <repo-url>
cd breast-cancer-predictor
2. Installer afhængigheder

Med conda:
conda create -n bc-predictor python=3.11 -y
conda activate bc-predictor
pip install -r requirements.txt
Med pip/venv:
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
requirements.txt indeholder:
pandas
numpy
scikit-learn
streamlit
joblib
matplotlib

3. Data

Projektet anvender Breast Cancer Wisconsin (Diagnostic) datasættet fra
UCI Machine Learning Repository
.

Hent wdbc.data og placer den i mappen:

breast+cancer+wisconsin+diagnostic/wdbc.data
4. Træn modeller

python train.py

Dette genererer models/summary.joblib og gemmer alle estimators i models/.

5. Start app

python -m streamlit run app.py
App’en åbner i browseren (http://localhost:8501
).

I notebooken train.py.ipynb findes en kort analyse af datasættet:

Class balance

Korrelations-heatmap

Boxplots for top-features

Roadmap

Hyperparameter tuning (GridSearchCV / cross-validation)

Flere modeller (XGBoost, LightGBM)

SHAP/Explainable AI for feature-forklaring

Deployment på Streamlit Cloud

Udvidelse til flere “Medical ML apps” (fx Diabetes, Heart Disease)

Credits

Data: UCI Machine Learning Repository

Kode: MIT License

