#!/usr/bin/env python3
"""
HYBRID PREDICTOR - Over 2.5 + BTTS
Combines: LogisticRegression + RandomForest + XGBoost
"""

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import poisson
import joblib


class HybridPredictor:
    """
    Hybrid Over 2.5 + BTTS predictor
    """

    def __init__(self):
        self.over25_model = None
        self.scaler = None

    def build_over25_ensemble(self):
        """Build voting classifier for Over 2.5"""

        logreg = LogisticRegression(
            C=1.0, max_iter=500, random_state=42, class_weight='balanced'
        )

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10,
            random_state=42, class_weight='balanced', n_jobs=-1
        )

        xgb = XGBClassifier(
            n_estimators=150, max_depth=8, learning_rate=0.1,
            scale_pos_weight=1.5, random_state=42, eval_metric='logloss'
        )

        ensemble = VotingClassifier(
            estimators=[('logreg', logreg), ('rf', rf), ('xgb', xgb)],
            voting='soft',
            weights=[0.4, 0.35, 0.25]
        )

        self.over25_model = CalibratedClassifierCV(ensemble, method='sigmoid', cv=5)
        return self.over25_model

    def predict_over25(self, features):
        """Predict Over 2.5"""
        if self.over25_model is None:
            raise ValueError("Model not trained!")

        proba = self.over25_model.predict_proba(features)[0, 1]
        lower = max(0.0, proba - 0.08)
        upper = min(1.0, proba + 0.08)

        if proba >= 0.75:
            confidence = "High"
        elif proba >= 0.65:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            'probability': float(proba),
            'confidence': confidence,
            'lower_bound': float(lower),
            'upper_bound': float(upper)
        }

    def predict_btts(self, lambda_home, lambda_away, context_multiplier=1.0):
        """Adaptive Poisson for BTTS"""
        p_home_scores = 1 - poisson.pmf(0, lambda_home * context_multiplier)
        p_away_scores = 1 - poisson.pmf(0, lambda_away * context_multiplier)
        p_btts = p_home_scores * p_away_scores

        if lambda_home < 1.0 or lambda_away < 1.0:
            p_btts *= 0.92

        return float(p_btts)

    def save(self, path="pretrained/"):
        """Save trained models"""
        joblib.dump(self.over25_model, f"{path}/over25_voting.pkl")
        joblib.dump(self.scaler, f"{path}/feature_scaler.pkl")

    def load(self, path="pretrained/"):
        """Load pre-trained models"""
        try:
            self.over25_model = joblib.load(f"{path}/over25_voting.pkl")
            self.scaler = joblib.load(f"{path}/feature_scaler.pkl")
        except FileNotFoundError:
            print("[WARN] Pre-trained models not found. Train first!")


if __name__ == "__main__":
    predictor = HybridPredictor()
    predictor.build_over25_ensemble()
    print("✓ Hybrid Predictor initialized")
