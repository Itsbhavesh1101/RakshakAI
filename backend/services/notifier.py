import asyncio
import base64
import html
import os
import smtplib
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Callable, Dict

import requests


class NotifierService:
    def __init__(self):
        # Configure variables (load from env or config overrides)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "moondream")

        self.smtp_enabled = os.getenv("SMTP_ENABLED", "false").lower() == "true"
        self.smtp_server = os.getenv("SMTP_SERVER", "")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_address = os.getenv("FROM_ADDRESS", "security-alert@platform.com")
        self.to_address = os.getenv("TO_ADDRESS", "")
        self.use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

        # Twilio SMS Integration Parameters
        self.sms_enabled = os.getenv("SMS_ENABLED", "false").lower() == "true"
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_from_number = os.getenv("TWILIO_FROM_NUMBER", "")
        self.to_phone = os.getenv("TO_PHONE", "")

        # Cooldown management per alert type to prevent duplicate notifications.
        self.email_cooldowns: Dict[str, float] = {}
        self.cooldown_seconds = float(os.getenv("EMAIL_COOLDOWN_SECONDS", "60"))
        self.http = requests.Session()

    def enrich_alert_with_ollama(self, alert_type: str, frame_jpeg: bytes) -> str:
        """
        Interrogates the local Ollama vision LLM for a natural language description.
        Runs inside a thread pool to protect FastAPI from blocking operations.
        """
        try:
            image_b64 = base64.b64encode(frame_jpeg).decode("utf-8")
            prompt = (
                "You are a smart surveillance platform security camera. "
                f"The alert system detected a '{alert_type}' threat in this frame. "
                "Describe the threat, location, people involved, urgency, and "
                "recommended security response actions."
            )

            payload = {
                "model": self.ollama_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 200,
            }

            url = f"{self.ollama_base_url}/v1/chat/completions"
            print(f"Ollama Request: Querying '{self.ollama_model}' for '{alert_type}' alert context...")
            response = self.http.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=(2, 12),
            )

            if response.status_code == 200:
                data = response.json()
                result_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not result_text:
                    raise ValueError("Ollama response did not include message content")
                print(f"Ollama Enrichment Complete: {result_text[:80]}...")
                return result_text

            print(f"Ollama API returned error {response.status_code}: {response.text}")
            return (
                "Standard threat event: "
                f"{alert_type.replace('_', ' ').title()} was detected by computer vision models."
            )
        except Exception as exc:
            print(f"Ollama connection skipped/failed: {exc}")
            return (
                "Computer vision threat triggers visual alarm validation for: "
                f"{alert_type.replace('_', ' ').title()}."
            )

    def send_smtp_email(self, alert_type: str, severity: str, description: str, frame_jpeg: bytes):
        """Dispatches email alerts with a JPEG snapshot attachment via SMTP."""
        if not self.smtp_enabled or not self.to_address:
            print(f"Email notification skipped (disabled or destination unconfigured). Alert type: {alert_type}")
            return

        current_time = time.time()
        last_sent = self.email_cooldowns.get(alert_type, 0.0)
        if current_time - last_sent < self.cooldown_seconds:
            print(f"Email notification throttled for '{alert_type}' to prevent mailbox overflow.")
            return

        self.email_cooldowns[alert_type] = current_time

        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            safe_description = html.escape(description)

            msg = MIMEMultipart()
            msg["Subject"] = f"[{severity}] Smart Surveillance Platform - {alert_type.replace('_', ' ').upper()}"
            msg["From"] = self.from_address
            msg["To"] = self.to_address

            body_html = f"""
            <html>
              <body style="font-family: Arial, sans-serif; background-color: #111; color: #eee; padding: 20px;">
                <div style="border: 2px solid #ff4444; border-radius: 8px; padding: 15px; background-color: #1a1a1a;">
                  <h2 style="color: #ff4444; margin-top: 0;">CRITICAL THREAT DETECTED</h2>
                  <p><strong>Incident Category:</strong> {alert_type.replace('_', ' ').title()}</p>
                  <p><strong>Priority Level:</strong> <span style="background-color: #ff4444; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{severity}</span></p>
                  <p><strong>Timestamp:</strong> {timestamp}</p>
                  <p><strong>AI Context Assessment:</strong> {safe_description}</p>
                  <hr style="border-color: #444;"/>
                  <p style="font-size: 12px; color: #888;">This is an automated dispatch from the Smart Surveillance Platform operations center. Please check the dashboard stream immediately.</p>
                </div>
              </body>
            </html>
            """
            msg.attach(MIMEText(body_html, "html"))

            img_attachment = MIMEImage(frame_jpeg)
            img_attachment.add_header("Content-Disposition", "attachment", filename=f"threat_{alert_type}.jpg")
            msg.attach(img_attachment)

            print(f"SMTP Alert sending to '{self.to_address}' via {self.smtp_server}:{self.smtp_port}...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            print(f"SMTP notification successfully dispatched for: '{alert_type}'")
        except Exception as exc:
            print(f"SMTP routing failed: {exc}")

    def send_twilio_sms(self, alert_type: str, severity: str, description: str):
        """Dispatches an SMS alert containing context and recommendation via Twilio API."""
        if not self.sms_enabled or not self.to_phone:
            print(f"SMS notification skipped (disabled or destination unconfigured). Alert: {alert_type}")
            return

        current_time = time.time()
        last_sent = self.email_cooldowns.get(f"sms_{alert_type}", 0.0)
        if current_time - last_sent < self.cooldown_seconds:
            print(f"SMS notification throttled for '{alert_type}' to prevent phone flooding.")
            return
        self.email_cooldowns[f"sms_{alert_type}"] = current_time

        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            body = (
                f"RAKSHAK ALERT: {severity} THREAT\n"
                f"Type: {alert_type.replace('_', ' ').upper()}\n"
                f"Time: {timestamp}\n"
                f"Assessment: {description}\n"
                f"Action Required: Please login to the security dashboard immediately."
            )

            # Check if Twilio credentials exist. If not, log to stdout as mock dispatch.
            if not self.twilio_account_sid or not self.twilio_auth_token or not self.twilio_from_number:
                print("\n" + "="*50)
                print("MOCK SMS DISPATCH (Twilio credentials unconfigured in .env)")
                print(f"To: {self.to_phone}")
                print(f"Body:\n{body}")
                print("="*50 + "\n")
                return

            print(f"Sending Twilio SMS alert to '{self.to_phone}'...")
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/Messages.json"
            data = {
                "From": self.twilio_from_number,
                "To": self.to_phone,
                "Body": body
            }
            response = self.http.post(
                url, 
                data=data, 
                auth=(self.twilio_account_sid, self.twilio_auth_token),
                timeout=8
            )

            if response.status_code in {200, 201}:
                print(f"Twilio SMS alert successfully sent! Message SID: {response.json().get('sid')}")
            else:
                print(f"Twilio SMS API returned error {response.status_code}: {response.text}")

        except Exception as exc:
            print(f"Twilio SMS dispatch failed: {exc}")

    async def schedule_notification_dispatch(
        self,
        alert_type: str,
        severity: str,
        frame_jpeg: bytes,
        callback_on_enrichment: Callable[[str], None],
    ):
        """
        Schedules Ollama description generation and SMS/email dispatches asynchronously.
        Uses run_in_executor to avoid blocking the active async loop during I/O.
        """
        loop = asyncio.get_running_loop()

        description = await loop.run_in_executor(
            None,
            self.enrich_alert_with_ollama,
            alert_type,
            frame_jpeg,
        )

        try:
            callback_on_enrichment(description)
        except Exception as exc:
            print(f"Alert enrichment callback failed: {exc}")

        if severity == "CRITICAL":
            if self.smtp_enabled:
                await loop.run_in_executor(
                    None,
                    self.send_smtp_email,
                    alert_type,
                    severity,
                    description,
                    frame_jpeg,
                )
            if self.sms_enabled:
                await loop.run_in_executor(
                    None,
                    self.send_twilio_sms,
                    alert_type,
                    severity,
                    description,
                )


notifier_service = NotifierService()
