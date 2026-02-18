#!/usr/bin/env python3
"""
Real-Time Monitor - Continuous 24/7 worker
Monitors upcoming matches and generates predictions
"""

import asyncio
import aiohttp
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.feature_engineering import FeatureEngineer
from models.hybrid_predictor import HybridPredictor
from core.database import Database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("monitor")


class RealTimeMonitor:
    """
    Continuous monitoring worker
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.api_key = self.config['api_keys']['football_data_org']
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {"X-Auth-Token": self.api_key}

        self.feature_engineer = FeatureEngineer()
        self.predictor = HybridPredictor()
        self.predictor.load()  # Load pre-trained model

        self.db = Database()
        self.running = True

    async def monitor_loop(self):
        """
        Main monitoring loop - runs forever
        """
        logger.info("="*70)
        logger.info("Real-Time Monitor started")
        logger.info("="*70)

        while self.running:
            try:
                await self.scan_upcoming_matches()

                # Sleep for configured interval
                interval = self.config['monitoring']['scan_interval_seconds']
                logger.info(f"Sleeping for {interval}s...")
                await asyncio.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                self.running = False
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(60)  # Wait 1 min on error

    async def scan_upcoming_matches(self):
        """
        Scan all leagues for upcoming matches
        """
        logger.info("Scanning upcoming matches...")

        now_utc = datetime.now(timezone.utc)
        date_from = now_utc.date().isoformat()
        look_ahead = self.config['monitoring']['look_ahead_days']
        date_to = (now_utc.date() + timedelta(days=look_ahead)).isoformat()

        async with aiohttp.ClientSession() as session:
            for code, info in self.config['leagues'].items():
                comp_id = info['id']
                comp_name = info['name']

                logger.info(f"  → {comp_name}")

                matches = await self._fetch_matches(session, comp_id, date_from, date_to)

                for match in matches:
                    if match.get('status') not in ('SCHEDULED', 'TIMED'):
                        continue

                    try:
                        kickoff = datetime.fromisoformat(match['utcDate'].replace('Z', '+00:00'))
                        if kickoff <= now_utc:
                            continue  # Skip past matches

                        # Check if already predicted
                        fixture_id = match['id']
                        if self.db.prediction_exists(fixture_id):
                            continue

                        # Generate prediction
                        prediction = await self._predict_match(session, match, comp_name)

                        if prediction:
                            self.db.save_prediction(prediction)
                            logger.info(f"    ✓ {prediction['home_team']} vs {prediction['away_team']}: "
                                      f"Over2.5={prediction['over25_prob']:.2f}, "
                                      f"BTTS={prediction['btts_prob']:.2f}")

                        # Rate limiting
                        delay = self.config['monitoring']['rate_limit_delay']
                        await asyncio.sleep(delay)

                    except Exception as e:
                        logger.error(f"    Error processing match: {e}")
                        continue

        logger.info("Scan complete\n")

    async def _fetch_matches(self, session, comp_id, date_from, date_to):
        """Fetch matches from football-data.org"""
        url = f"{self.base_url}/competitions/{comp_id}/matches"
        params = {'dateFrom': date_from, 'dateTo': date_to}

        try:
            async with session.get(url, headers=self.headers, params=params, timeout=20) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('matches', [])
                else:
                    logger.warning(f"API status {resp.status} for comp {comp_id}")
                    return []
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return []

    async def _fetch_team_history(self, session, team_id, limit=10):
        """Fetch team's recent matches"""
        url = f"{self.base_url}/teams/{team_id}/matches"
        params = {'status': 'FINISHED', 'limit': limit}

        try:
            async with session.get(url, headers=self.headers, params=params, timeout=20) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('matches', [])
                return []
        except Exception:
            return []

    async def _predict_match(self, session, match, league):
        """
        Generate prediction for a match
        """
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        home_id = match['homeTeam']['id']
        away_id = match['awayTeam']['id']
        kickoff = match['utcDate']

        # Fetch history
        home_history = await self._fetch_team_history(session, home_id, 15)
        away_history = await self._fetch_team_history(session, away_id, 15)

        if not home_history or not away_history:
            return None

        # Extract features
        features_dict = self.feature_engineer.extract_features(
            home_team, away_team,
            home_history, away_history,
            league,
            datetime.fromisoformat(kickoff.replace('Z', '+00:00'))
        )

        # Convert to numpy array (36 features)
        import numpy as np
        feature_vector = np.array([list(features_dict.values())])

        # Predict Over 2.5
        over25_result = self.predictor.predict_over25(feature_vector)

        # Predict BTTS
        btts_prob = self.predictor.predict_btts(
            features_dict['lambda_home'],
            features_dict['lambda_away'],
            context_multiplier=1.0
        )

        # Apply thresholds
        min_over25 = self.config['thresholds']['over25_min_probability']
        min_btts = self.config['thresholds']['btts_min_probability']

        if over25_result['probability'] < min_over25:
            return None  # Filter out low probability

        return {
            'fixture_id': match['id'],
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'kickoff_utc': kickoff,
            'over25_prob': over25_result['probability'],
            'over25_confidence': over25_result['confidence'],
            'btts_prob': btts_prob,
            'expected_goals': features_dict['expected_total_goals'],
            'home_form': features_dict['home_points_form_5'],
            'away_form': features_dict['away_points_form_3']
        }


def main():
    monitor = RealTimeMonitor()
    asyncio.run(monitor.monitor_loop())


if __name__ == "__main__":
    main()
