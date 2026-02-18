#!/usr/bin/env python3
"""
Database module - SQLite for predictions storage
"""

import sqlite3
from pathlib import Path
from datetime import datetime


class Database:
    """
    SQLite database for real-time predictions
    """

    def __init__(self, db_path="data/predictions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id INTEGER UNIQUE,
                predicted_at TIMESTAMP,
                home_team TEXT,
                away_team TEXT,
                league TEXT,
                kickoff_utc TIMESTAMP,
                over25_prob REAL,
                over25_confidence TEXT,
                btts_prob REAL,
                expected_goals REAL,
                home_form REAL,
                away_form REAL,
                status TEXT DEFAULT 'PENDING'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id INTEGER,
                home_goals INTEGER,
                away_goals INTEGER,
                over25_actual INTEGER,
                btts_actual INTEGER,
                recorded_at TIMESTAMP,
                FOREIGN KEY (fixture_id) REFERENCES predictions(fixture_id)
            )
        """)

        conn.commit()
        conn.close()

    def save_prediction(self, prediction):
        """Save a new prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO predictions
            (fixture_id, predicted_at, home_team, away_team, league, kickoff_utc,
             over25_prob, over25_confidence, btts_prob, expected_goals, home_form, away_form)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction['fixture_id'],
            datetime.now().isoformat(),
            prediction['home_team'],
            prediction['away_team'],
            prediction['league'],
            prediction['kickoff_utc'],
            prediction['over25_prob'],
            prediction['over25_confidence'],
            prediction['btts_prob'],
            prediction['expected_goals'],
            prediction['home_form'],
            prediction['away_form']
        ))

        conn.commit()
        conn.close()

    def prediction_exists(self, fixture_id):
        """Check if prediction already exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE fixture_id = ?", (fixture_id,))
        exists = cursor.fetchone()[0] > 0

        conn.close()
        return exists

    def get_active_predictions(self, min_prob=0.65):
        """Get all future predictions above threshold"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM predictions
            WHERE over25_prob >= ?
            AND kickoff_utc > datetime('now')
            ORDER BY over25_prob DESC
        """, (min_prob,))

        rows = cursor.fetchall()
        conn.close()

        return rows

    def save_result(self, fixture_id, home_goals, away_goals):
        """Save actual match result"""
        over25 = 1 if (home_goals + away_goals) > 2 else 0
        btts = 1 if (home_goals > 0 and away_goals > 0) else 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO results (fixture_id, home_goals, away_goals, over25_actual, btts_actual, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (fixture_id, home_goals, away_goals, over25, btts, datetime.now().isoformat()))

        cursor.execute("UPDATE predictions SET status = 'FINISHED' WHERE fixture_id = ?", (fixture_id,))

        conn.commit()
        conn.close()

    def get_accuracy_stats(self, days=7):
        """Calculate model accuracy for last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN (r.over25_actual = 1 AND p.over25_prob >= 0.65) OR
                             (r.over25_actual = 0 AND p.over25_prob < 0.65) THEN 1 ELSE 0 END) as correct_over25,
                SUM(CASE WHEN (r.btts_actual = 1 AND p.btts_prob >= 0.60) OR
                             (r.btts_actual = 0 AND p.btts_prob < 0.60) THEN 1 ELSE 0 END) as correct_btts
            FROM predictions p
            JOIN results r ON p.fixture_id = r.fixture_id
            WHERE p.predicted_at >= datetime('now', '-' || ? || ' days')
        """, (days,))

        row = cursor.fetchone()
        conn.close()

        if row[0] == 0:
            return {'accuracy_over25': 0.0, 'accuracy_btts': 0.0, 'total': 0}

        return {
            'accuracy_over25': row[1] / row[0] if row[0] > 0 else 0.0,
            'accuracy_btts': row[2] / row[0] if row[0] > 0 else 0.0,
            'total': row[0]
        }


if __name__ == "__main__":
    db = Database()
    print("✓ Database initialized")
