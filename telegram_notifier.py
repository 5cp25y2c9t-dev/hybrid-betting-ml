#!/usr/bin/env python3
"""
Telegram Notifier - Optional instant alerts
"""

import asyncio
import aiohttp
import logging

logger = logging.getLogger("telegram")


class TelegramNotifier:
    def __init__(self, token, chat_id, enabled=True):
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled and token and chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    async def send_message(self, text):
        if not self.enabled:
            return

        url = f"{self.base_url}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        logger.info("✓ Telegram sent")
                    else:
                        logger.warning(f"Telegram error: {resp.status}")
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    async def send_prediction_alert(self, prediction):
        """Send alert for high-value prediction"""
        if not self.enabled:
            return

        if prediction['over25_prob'] < 0.70:
            return  # Only high-confidence

        text = (
            f"🎯 <b>HIGH VALUE BET</b>\n\n"
            f"⚽ {prediction['home_team']} vs {prediction['away_team']}\n"
            f"🏆 {prediction['league']}\n"
            f"🕐 {prediction['kickoff_utc']}\n\n"
            f"📊 Over 2.5: <b>{prediction['over25_prob']*100:.1f}%</b>\n"
            f"🎲 BTTS: {prediction['btts_prob']*100:.1f}%\n"
            f"⚡ Confidence: {prediction['over25_confidence']}"
        )

        await self.send_message(text)


if __name__ == "__main__":
    notifier = TelegramNotifier("TOKEN", "CHAT_ID", enabled=False)
    print("✓ Telegram Notifier ready")
