# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-FileContributor: Amit Bhakt <amitbhakt@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""
Configuration models for SDialog WhatsApp server.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WhatsAppConfig:
    """
    Configuration parameters for serving an SDialog agent on WhatsApp.

    Parameters will fall back to standard environment variables if not provided:
    - ``WHATSAPP_PHONE_NUMBER_ID``
    - ``WHATSAPP_ACCESS_TOKEN``
    - ``WHATSAPP_VERIFY_TOKEN``
    - ``WHATSAPP_APP_SECRET``
    """

    port: int = 1234
    host: str = "0.0.0.0"
    phone_number_id: Optional[str] = field(
        default_factory=lambda: os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
    )
    access_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("WHATSAPP_ACCESS_TOKEN") or os.environ.get("WHATSAPP_TOKEN")
    )
    verify_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("WHATSAPP_VERIFY_TOKEN")
        or os.environ.get("WHATSAPP_WEBHOOK_VERIFY_TOKEN", "sdialog_whatsapp_secret")
    )
    app_secret: Optional[str] = field(
        default_factory=lambda: os.environ.get("WHATSAPP_APP_SECRET")
    )
    api_version: str = "v21.0"
    db_path: str = "whatsapp_sessions.db"
    webhook_path: str = "/webhook"
    session_timeout: int = 1800  # 30 minutes idle timeout
    tunnel: bool = False
    auto_setup: bool = True  # Automatically discover IDs and configure Meta webhooks
    max_history_turns: int = 40  # Context window sliding budget
    enforce_signature: bool = True
    mark_as_read: bool = True
    log_level: str = "info"

    def validate(self, dry_run: bool = False) -> None:
        """Validate that required credentials are set when not in dry_run mode."""
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
