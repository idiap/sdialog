# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Amit Bhakt <amitbhakt@users.noreply.github.com>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
"""
Meta WhatsApp Cloud API client.
"""

import logging
from typing import List
import httpx

from .config import WhatsAppConfig

logger = logging.getLogger(__name__)

# WhatsApp limits single message text body to 4096 characters
MAX_WHATSAPP_TEXT_LENGTH = 4096


class MetaWhatsAppClient:
    """Async client for interacting with the Meta WhatsApp Cloud API."""

    def __init__(self, config: WhatsAppConfig):
        self.config = config
        self.base_url = f"https://graph.facebook.com/{config.api_version}/{config.phone_number_id}"
        self.headers = {
            "Authorization": f"Bearer {config.access_token}",
            "Content-Type": "application/json",
        }

    async def send_text_message(self, to: str, text: str) -> bool:
        """
        Send a text message to a WhatsApp user.

        If the text exceeds WhatsApp's maximum length (4096 characters),
        it is split and sent in chunks.
        """
        if not text:
            return False

        chunks = self._chunk_text(text, MAX_WHATSAPP_TEXT_LENGTH)
        success = True

        async with httpx.AsyncClient(timeout=10.0) as client:
            for chunk in chunks:
                payload = {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": to,
                    "type": "text",
                    "text": {"preview_url": False, "body": chunk},
                }

                try:
                    response = await client.post(
                        f"{self.base_url}/messages",
                        json=payload,
                        headers=self.headers,
                    )

                    if response.status_code in (200, 201):
                        logger.debug(f"[WhatsApp] Sent message to {to}")
                    else:
                        logger.error(
                            f"[WhatsApp] Failed to send message to {to}. "
                            f"HTTP {response.status_code}: {response.text}"
                        )
                        success = False
                except Exception as e:
                    logger.error(f"[WhatsApp] Exception while sending message to {to}: {e}")
                    success = False

        return success

    async def mark_as_read(self, message_id: str) -> bool:
        """Mark an incoming WhatsApp message as read (blue double check)."""
        if not message_id:
            return False

        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=self.headers,
                )
                return response.status_code == 200
            except Exception as e:
                logger.debug(f"[WhatsApp] Could not mark message {message_id} as read: {e}")
                return False

    @staticmethod
    def _chunk_text(text: str, max_length: int) -> List[str]:
        """Split long messages into WhatsApp-safe chunks without splitting words where possible."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break

            # Find last whitespace within max_length
            split_idx = text.rfind("\n", 0, max_length)
            if split_idx == -1:
                split_idx = text.rfind(" ", 0, max_length)
            if split_idx == -1:
                split_idx = max_length

            chunks.append(text[:split_idx].strip())
            text = text[split_idx:].lstrip()

        return chunks
