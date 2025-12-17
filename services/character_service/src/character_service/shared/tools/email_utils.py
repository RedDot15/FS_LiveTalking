import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime

import emails
from jinja2 import Template

from character_service.shared.utils import get_settings

from mongo_client.model.entity import Request

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmailData:
    html_content: str
    subject: str


def render_email_template(*, template_name: str, context: dict[str, Any]) -> str:
    template_str = (
        Path(__file__).parent / "email-templates" / "build" / template_name
    ).read_text()
    html_content = Template(template_str).render(context)
    return html_content

def send_email(
    *,
    email_to: str,
    subject: str = "",
    html_content: str = "",
) -> None:
    assert settings.emails_enabled, "no provided configuration for email variables"
    message = emails.Message(
        subject=subject,
        html=html_content,
        mail_from=(settings.EMAILS_FROM_NAME, settings.EMAILS_FROM_EMAIL),
    )
    smtp_options = {"host": settings.SMTP_HOST, "port": settings.SMTP_PORT}
    if settings.SMTP_TLS:
        smtp_options["tls"] = True
    elif settings.SMTP_SSL:
        smtp_options["ssl"] = True
    if settings.SMTP_USER:
        smtp_options["user"] = settings.SMTP_USER
    if settings.SMTP_PASSWORD:
        smtp_options["password"] = settings.SMTP_PASSWORD
    response = message.send(to=email_to, smtp=smtp_options)
    logger.info(f"send email result: {response}")

def generate_request_approved_email(
    request: Request
) -> EmailData:
    project_name = settings.PROJECT_NAME
    subject = f"{project_name} - Request approved"
    html_content = render_email_template(
        template_name="request_approved.html",
        context={
            "character_name": request['character_name'],
            "character_id": request['character_id'],
            "character_url": f"{settings.FRONTEND_HOST}/character/{request['character_id']}",
            "created_at": request['created_at'],
            "created_by": request['created_by'],
            "approved_at": datetime.now(),
            "approved_by": request['approved_by'],
        },
    )
    return EmailData(html_content=html_content, subject=subject)