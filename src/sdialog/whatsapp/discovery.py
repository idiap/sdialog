# SPDX-FileCopyrightText: Copyright © 2026 Amit Bhakt <https://github.com/amitbhakt/wavoice>
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-License-Identifier: MIT
#
# Portions of this module are adapted from WAVoice:
# "WAVoice: WebRTC Gateway and DevOps Automation Engine for WhatsApp Voice AI"
# Repository: https://github.com/amitbhakt/wavoice
# DOI: 10.5281/zenodo.23537489
# Author: Amit Bhakt <yashubhakt@gmail.com>
# Licensed under the MIT License.
"""
Meta WhatsApp Cloud API auto-discovery and multi-tier webhook configuration.

This module automates the onboarding process for WhatsApp Cloud API agents:
1. Token inspection (extracts App ID, WABA ID, permissions)
2. Automatic discovery of Phone Number ID and WABA ID
3. 3-Tier Webhook synchronization (App-level, WABA-level, and Phone-number-level)
4. Detection of welcome messages (``request_welcome`` webhooks on new chats)
"""

import logging
from typing import Optional, Dict, Any, List
import httpx

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v21.0"
GRAPH_BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


class MetaDiscoveryClient:
    """
    Client for automating Meta WhatsApp Cloud API credentials discovery and webhook setup.

    Adapted from WAVoice (MIT License, https://github.com/amitbhakt/wavoice).
    """

    def __init__(self, access_token: str, app_secret: Optional[str] = None):
        self.access_token = access_token
        self.app_secret = app_secret

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def inspect_token(self) -> Dict[str, Any]:
        """
        Inspect the provided User/System Access Token via debug_token endpoint.

        Returns metadata including app_id, scopes, user_id, and granular target IDs (WABAs).
        """
        url = f"{GRAPH_BASE_URL}/debug_token"
        params = {"input_token": self.access_token, "access_token": self.access_token}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
            if resp.status_code != 200:
                error_msg = data.get("error", {}).get("message", "Token inspection failed")
                raise ValueError(f"Invalid Meta Access Token: {error_msg}")
            return data.get("data", {})

    async def discover_waba_id(self) -> Optional[str]:
        """Auto-discover WhatsApp Business Account (WABA) ID from token permissions."""
        try:
            token_data = await self.inspect_token()
            granular_scopes = token_data.get("granular_scopes", [])
            for scope_info in granular_scopes:
                if scope_info.get("scope") in ("whatsapp_business_management", "whatsapp_business_messaging"):
                    target_ids = scope_info.get("target_ids", [])
                    if target_ids:
                        return str(target_ids[0])
        except Exception as e:
            logger.debug(f"[WhatsApp] Could not auto-discover WABA ID: {e}")
        return None

    async def list_phone_numbers(self, waba_id: str) -> List[Dict[str, Any]]:
        """List all phone numbers under a WhatsApp Business Account."""
        url = f"{GRAPH_BASE_URL}/{waba_id}/phone_numbers"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=self._headers)
            if resp.status_code == 200:
                return resp.json().get("data", [])
        return []

    async def discover_phone_number_id(self, waba_id: Optional[str] = None) -> Optional[str]:
        """Auto-discover the primary WhatsApp Phone Number ID."""
        target_waba = waba_id or await self.discover_waba_id()
        if not target_waba:
            return None

        numbers = await self.list_phone_numbers(target_waba)
        if numbers:
            phone_id = str(numbers[0].get("id"))
            display_num = numbers[0].get("display_phone_number", "unknown")
            logger.info(f"[WhatsApp] Auto-discovered Phone Number ID: {phone_id} ({display_num})")
            return phone_id
        return None

    async def is_welcome_message_enabled(self, phone_number_id: str) -> Optional[bool]:
        """
        Check whether welcome messages are enabled on a phone number, in which case Meta sends a
        ``request_welcome`` webhook whenever a user opens a new chat with it.

        Returns None if the setting could not be read.
        """
        url = f"{GRAPH_BASE_URL}/{phone_number_id}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=self._headers, params={"fields": "conversational_automation"})
            if resp.status_code != 200:
                logger.debug(f"[WhatsApp] Could not read conversational_automation: {resp.text}")
                return None
            # The field is omitted when nothing was ever configured on the number
            automation = resp.json().get("conversational_automation", {})
            return bool(automation.get("enable_welcome_message", False))

    async def configure_webhooks(
        self,
        callback_url: str,
        verify_token: str,
        phone_number_id: Optional[str] = None,
        waba_id: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Programmatically synchronize all 3 tiers of Meta WhatsApp webhooks:
        1. App-level subscription (POST /{app_id}/subscriptions)
        2. WABA-level subscription (POST /{waba_id}/subscribed_apps)
        3. Phone-number-level override (POST /{phone_number_id})
        """
        results = {"app_level": False, "waba_level": False, "phone_level": False}

        token_info = await self.inspect_token()
        app_id = token_info.get("app_id")
        target_waba = waba_id or await self.discover_waba_id()

        async with httpx.AsyncClient(timeout=15.0) as client:
            # Tier 1: App-Level Webhook Subscription (requires App Access Token)
            if app_id and self.app_secret:
                try:
                    app_access_token = f"{app_id}|{self.app_secret}"
                    app_sub_url = f"{GRAPH_BASE_URL}/{app_id}/subscriptions"
                    payload = {
                        "object": "whatsapp_business_account",
                        "callback_url": callback_url,
                        "verify_token": verify_token,
                        "fields": "messages",
                        "include_values": "true",
                        "access_token": app_access_token,
                    }
                    resp = await client.post(app_sub_url, data=payload)
                    if resp.status_code == 200 and resp.json().get("success"):
                        results["app_level"] = True
                        logger.info("[WhatsApp] Configured App-level webhook subscription")
                    else:
                        logger.warning(f"[WhatsApp] App-level webhook config returned: {resp.text}")
                except Exception as e:
                    logger.warning(f"[WhatsApp] Could not set App-level webhook: {e}")

            # Tier 2: WABA-Level Subscribed Apps
            if target_waba:
                try:
                    waba_sub_url = f"{GRAPH_BASE_URL}/{target_waba}/subscribed_apps"
                    waba_payload = {
                        "override_callback_uri": callback_url,
                        "verify_token": verify_token,
                    }
                    resp = await client.post(waba_sub_url, headers=self._headers, json=waba_payload)
                    if resp.status_code == 200 and resp.json().get("success"):
                        results["waba_level"] = True
                        logger.info(f"[WhatsApp] Configured WABA-level webhook override on {target_waba}")
                    else:
                        logger.warning(f"[WhatsApp] WABA-level webhook config returned: {resp.text}")
                except Exception as e:
                    logger.warning(f"[WhatsApp] Could not set WABA-level webhook: {e}")

            # Tier 3: Phone-Number-Level Webhook Override
            target_phone = phone_number_id or await self.discover_phone_number_id(target_waba)
            if target_phone:
                try:
                    phone_url = f"{GRAPH_BASE_URL}/{target_phone}"
                    phone_payload = {
                        "webhook_configuration": {
                            "override_callback_uri": callback_url,
                            "verify_token": verify_token,
                        }
                    }
                    resp = await client.post(phone_url, headers=self._headers, json=phone_payload)
                    if resp.status_code == 200 and resp.json().get("success"):
                        results["phone_level"] = True
                        logger.info(f"[WhatsApp] Configured Phone-level webhook override on {target_phone}")
                    else:
                        logger.warning(f"[WhatsApp] Phone-level webhook config returned: {resp.text}")
                except Exception as e:
                    logger.warning(f"[WhatsApp] Could not set Phone-level webhook: {e}")

        return results
