# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-FileContributor: Amit Bhakt <amitbhakt@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""
Configuration models for SDialog WhatsApp server.
"""

import os
import secrets
from dataclasses import dataclass
from typing import Optional


@dataclass
class WhatsAppConfig:
    """
    Configuration parameters for serving an SDialog agent on WhatsApp.

    Parameters will fall back to standard environment variables if not provided:
    - ``WHATSAPP_PHONE_NUMBER_ID``
    - ``WHATSAPP_ACCESS_TOKEN``
    - ``WHATSAPP_VERIFY_TOKEN`` (a random token is generated if not set)
    - ``WHATSAPP_APP_SECRET`` (required unless ``enforce_signature=False``)
    """

    port: int = 1234
    host: str = "0.0.0.0"
    phone_number_id: Optional[str] = None
    access_token: Optional[str] = None
    verify_token: Optional[str] = None
    app_secret: Optional[str] = None
    api_version: str = "v21.0"
    db_path: str = "whatsapp_sessions.db"
    webhook_path: str = "/webhook"
    session_timeout: int = 1800  # 30 minutes idle timeout
    tunnel: bool = False
    # Overwrites the webhook URL of the business account and phone number, use only with test accounts
    auto_setup: bool = False
    max_history_turns: int = 40  # Context window sliding budget
    enforce_signature: bool = True  # Reject webhook requests not signed with app_secret
    mark_as_read: bool = True
    log_level: str = "info"

    def __post_init__(self):
        # Resolved here rather than with default_factory so that explicit None values also fall back to env vars
        self.phone_number_id = self.phone_number_id or os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
        self.access_token = (self.access_token
                             or os.environ.get("WHATSAPP_ACCESS_TOKEN") or os.environ.get("WHATSAPP_TOKEN"))
        self.verify_token = (self.verify_token
                             or os.environ.get("WHATSAPP_VERIFY_TOKEN")
                             or os.environ.get("WHATSAPP_WEBHOOK_VERIFY_TOKEN"))
        self.app_secret = self.app_secret or os.environ.get("WHATSAPP_APP_SECRET")

        self.verify_token_generated = not self.verify_token
        if self.verify_token_generated:
            self.verify_token = secrets.token_urlsafe(24)

    def validate(self, dry_run: bool = False) -> None:
        """Validate that required credentials are set when not in dry_run mode."""
        if self.enforce_signature and not self.app_secret:
            raise ValueError(
                "WhatsApp app_secret is required to verify that webhook requests come from Meta. "
                "Pass it to serve_whatsapp() or set the WHATSAPP_APP_SECRET environment variable "
                "(or set enforce_signature=False to accept unsigned requests, NOT recommended)."
            )
        if not dry_run:
            if not self.phone_number_id:
                raise ValueError(
                    "WhatsApp phone_number_id is required. Pass it to serve_whatsapp() "
                    "or set the WHATSAPP_PHONE_NUMBER_ID environment variable."
                )
            if not self.access_token:
                raise ValueError(
                    "WhatsApp access_token is required. Pass it to serve_whatsapp() "
                    "or set the WHATSAPP_ACCESS_TOKEN environment variable."
                )
