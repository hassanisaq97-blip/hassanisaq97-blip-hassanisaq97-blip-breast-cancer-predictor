import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DATABASE_PATH = Path("prediction_logs.db")


def initialize_database():
    """
    Opret tabellen til logning, hvis den ikke allerede findes.
    """
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prediction TEXT NOT NULL,
                malignancy_probability REAL NOT NULL,
                input_features TEXT NOT NULL,
                shap_explanation TEXT NOT NULL,
                ai_summary TEXT NOT NULL
            )
            """
        )


def log_prediction(
    model_name,
    prediction,
    probability,
    input_features,
    explanation_table,
    ai_summary,
):
    
    initialize_database()

    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO prediction_logs (
                created_at,
                model_name,
                prediction,
                malignancy_probability,
                input_features,
                shap_explanation,
                ai_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                str(model_name),
                str(prediction),
                float(probability),
                json.dumps(input_features, default=str),
                json.dumps(explanation_table, default=str),
                str(ai_summary),
            ),
        )
        connection.commit()
def get_prediction_history():

    initialize_database()

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                model_name,
                prediction,
                malignancy_probability,
                input_features,
                shap_explanation,
                ai_summary
            FROM prediction_logs
            ORDER BY id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]