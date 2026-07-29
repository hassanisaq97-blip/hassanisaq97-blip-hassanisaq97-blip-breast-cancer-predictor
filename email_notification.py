import smtplib
import ssl
from email.message import EmailMessage


def send_prediction_email(
    sender_email,
    sender_password,
    receiver_email,
    model_name,
    prediction,
    probability,
    shap_explanation,
    ai_summary,
):
    """
    Send en automatisk e-mail med prediction, SHAP-forklaring
    og AI-resumé.
    """

    subject = f"AI-notifikation: {prediction}"

    body = f"""
En ny prediction er blevet registreret.

Model:
{model_name}

Prediction:
{prediction}

Sandsynlighed for malignitet:
{probability * 100:.2f} %

SHAP-forklaring:
{shap_explanation}

AI-resumé:
{ai_summary}

Dette er en automatisk genereret besked fra et demonstrationsprojekt.
Systemet må ikke anvendes til kliniske beslutninger.
"""

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email
    message.set_content(body)

    secure_context = ssl.create_default_context()

    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465,
        context=secure_context,
    ) as smtp_server:
        smtp_server.login(sender_email, sender_password)
        smtp_server.send_message(message)