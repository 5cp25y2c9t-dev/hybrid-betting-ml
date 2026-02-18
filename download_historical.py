#!/usr/bin/env python3
"""
Download Historical Data
Downloads CSVs from football-data.co.uk for training
"""

import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download")


LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1"
}

SEASONS = ["2122", "2223", "2324"]  # 2021-22, 2022-23, 2023-24


def download_historical_data():
    """
    Download historical match data from football-data.co.uk
    """
    base_url = "https://www.football-data.co.uk/mmz4281"
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading historical data from football-data.co.uk...")
    logger.info("="*70)

    for season in SEASONS:
        for code, name in LEAGUES.items():
            url = f"{base_url}/{season}/{code}.csv"
            output_file = output_dir / f"{code}_{season}.csv"

            if output_file.exists():
                logger.info(f"  ✓ {name} {season} - already exists")
                continue

            try:
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"  ✓ {name} {season} - downloaded")
                else:
                    logger.warning(f"  ✗ {name} {season} - HTTP {response.status_code}")

            except Exception as e:
                logger.error(f"  ✗ {name} {season} - Error: {e}")

    logger.info("="*70)
    logger.info("✓ Download complete")

    # Count files
    csv_count = len(list(output_dir.glob("*.csv")))
    logger.info(f"Total CSV files: {csv_count}")


if __name__ == "__main__":
    download_historical_data()
