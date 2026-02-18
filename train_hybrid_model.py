#!/usr/bin/env python3
"""
Train Hybrid Model - Full training pipeline
Downloads historical data and trains ensemble model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging

from models.hybrid_predictor import HybridPredictor
from models.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")


def load_historical_data(data_path="data/raw"):
    """
    Load historical data (CSVs downloaded from football-data.co.uk)
    Expected format: Date,HomeTeam,AwayTeam,FTHG,FTAG
    """
    logger.info("Loading historical data...")

    dfs = []
    for csv_file in Path(data_path).glob("*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No CSV files found in data/raw/. Run download_historical.py first!")

    data = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(data)} matches")

    return data


def create_training_samples(data, feature_engineer):
    """
    Convert raw match data to feature vectors
    """
    logger.info("Creating training samples...")

    X = []
    y_over25 = []
    y_btts = []

    # Group by team to get history
    for idx, match in data.iterrows():
        if idx % 100 == 0:
            logger.info(f"  Processed {idx}/{len(data)}")

        # Skip if no goals data
        if pd.isna(match['FTHG']) or pd.isna(match['FTAG']):
            continue

        home_team = match['HomeTeam']
        away_team = match['AwayTeam']

        # Get historical matches (previous matches in dataset)
        home_history = data[(data['HomeTeam'] == home_team) | (data['AwayTeam'] == home_team)]
        away_history = data[(data['HomeTeam'] == away_team) | (data['AwayTeam'] == away_team)]

        # Take only previous matches
        home_history = home_history[home_history.index < idx].tail(15)
        away_history = away_history[away_history.index < idx].tail(15)

        if len(home_history) < 10 or len(away_history) < 10:
            continue  # Not enough history

        # Convert to format expected by FeatureEngineer
        home_hist_formatted = []
        for _, h in home_history.iterrows():
            home_hist_formatted.append({
                'homeTeam': {'name': h['HomeTeam']},
                'awayTeam': {'name': h['AwayTeam']},
                'score': {'fullTime': {'home': h['FTHG'], 'away': h['FTAG']}}
            })

        away_hist_formatted = []
        for _, a in away_history.iterrows():
            away_hist_formatted.append({
                'homeTeam': {'name': a['HomeTeam']},
                'awayTeam': {'name': a['AwayTeam']},
                'score': {'fullTime': {'home': a['FTHG'], 'away': a['FTAG']}}
            })

        # Extract features
        try:
            features = feature_engineer.extract_features(
                home_team, away_team,
                home_hist_formatted, away_hist_formatted,
                "Premier League",  # Assume PL for now
                pd.to_datetime(match['Date'])
            )

            X.append(list(features.values()))

            # Labels
            total_goals = match['FTHG'] + match['FTAG']
            y_over25.append(1 if total_goals > 2 else 0)
            y_btts.append(1 if (match['FTHG'] > 0 and match['FTAG'] > 0) else 0)

        except Exception as e:
            continue

    logger.info(f"Created {len(X)} training samples")

    return np.array(X), np.array(y_over25), np.array(y_btts)


def train_model(X, y):
    """
    Train the hybrid ensemble model
    """
    logger.info("="*70)
    logger.info("Training Hybrid Model (Over 2.5)")
    logger.info("="*70)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train model
    predictor = HybridPredictor()
    predictor.scaler = scaler
    model = predictor.build_over25_ensemble()

    logger.info("Training ensemble (this may take 10-15 minutes)...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    logger.info(f"Train accuracy: {train_score:.3f}")
    logger.info(f"Test accuracy:  {test_score:.3f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    logger.info(f"CV accuracy:    {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    predictor.over25_model = model

    return predictor


def main():
    # Load data
    data = load_historical_data()

    # Create features
    fe = FeatureEngineer()
    X, y_over25, y_btts = create_training_samples(data, fe)

    if len(X) < 100:
        logger.error("Not enough training samples! Need at least 100.")
        return

    # Train Over 2.5 model
    predictor = train_model(X, y_over25)

    # Save model
    Path("pretrained").mkdir(exist_ok=True)
    predictor.save("pretrained")

    logger.info("="*70)
    logger.info("✓ Model trained and saved to pretrained/")
    logger.info("="*70)


if __name__ == "__main__":
    main()
