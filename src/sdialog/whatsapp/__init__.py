# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Amit Bhakt <amitbhakt@users.noreply.github.com>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
"""
WhatsApp integration for SDialog agents.

This package provides native WhatsApp serving capabilities for SDialog agents,
enabling two-way text conversations over the Meta WhatsApp Cloud API with SQLite
persistence and per-user session isolation.
"""

from .config import WhatsAppConfig
from .server import WhatsAppServer
from .session_store import SQLiteSessionStore
from .client import MetaWhatsAppClient
from .discovery import MetaDiscoveryClient

__all__ = [
    "WhatsAppConfig",
    "WhatsAppServer",
    "SQLiteSessionStore",
    "MetaWhatsAppClient",
    "MetaDiscoveryClient",
]
