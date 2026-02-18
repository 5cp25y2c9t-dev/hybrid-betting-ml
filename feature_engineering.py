#!/usr/bin/env python3
"""
Feature Engineering - 36 features dla Over 2.5 + BTTS
Combines best practices from AIFootballPredictions + ProphitBet
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class FeatureEngineer:
    """
    Extract 36 features from match history
    """

    def __init__(self):
        self.league_avg_goals = {
            "Premier League": 2.82,
            "La Liga": 2.63,
            "Serie A": 2.89,
            "Bundesliga": 3.11,
            "Ligue 1": 2.71
        }

    def extract_features(self, home_team, away_team, home_history, away_history, 
                        league, kickoff_date):
        """
        Extract all 36 features for a match

        Returns:
            dict: 36 features
        """
        features = {}

        # BLOCK 1: Form & Momentum (7 features)
        features['home_goals_avg_5'] = self._goals_avg(home_history[:5], home_team)
        features['away_goals_avg_5'] = self._goals_avg(away_history[:5], away_team)
        features['home_goals_avg_10'] = self._goals_avg(home_history[:10], home_team)
        features['away_goals_avg_10'] = self._goals_avg(away_history[:10], away_team)

        features['home_points_form_3'] = self._points_form(home_history[:3], home_team)
        features['away_points_form_3'] = self._points_form(away_history[:3], away_team)
        features['home_points_form_5'] = self._points_form(home_history[:5], home_team)

        # BLOCK 2: Attack/Defense Strength (8 features)
        league_avg = self.league_avg_goals.get(league, 2.75)

        features['home_attack_strength'] = features['home_goals_avg_10'] / (league_avg / 2)
        features['away_attack_strength'] = features['away_goals_avg_10'] / (league_avg / 2)

        features['home_defense_strength'] = self._goals_conceded_avg(home_history[:10], home_team) / (league_avg / 2)
        features['away_defense_strength'] = self._goals_conceded_avg(away_history[:10], away_team) / (league_avg / 2)

        # Trends
        recent_5 = self._goals_avg(home_history[:5], home_team)
        previous_5 = self._goals_avg(home_history[5:10], home_team)
        features['home_goals_trend'] = recent_5 - previous_5

        recent_5_a = self._goals_avg(away_history[:5], away_team)
        previous_5_a = self._goals_avg(away_history[5:10], away_team)
        features['away_goals_trend'] = recent_5_a - previous_5_a

        features['home_conceded_trend'] = self._goals_conceded_avg(home_history[:5], home_team) - self._goals_conceded_avg(home_history[5:10], home_team)
        features['away_conceded_trend'] = self._goals_conceded_avg(away_history[:5], away_team) - self._goals_conceded_avg(away_history[5:10], away_team)

        # BLOCK 3: Match Context (8 features)
        features['home_advantage'] = 1.0
        features['league_avg_goals'] = league_avg
        features['home_rest_days'] = 7
        features['away_rest_days'] = 7

        h2h_goals = self._h2h_goals_avg(home_team, away_team, home_history + away_history)
        features['h2h_goals_avg'] = h2h_goals
        features['h2h_btts_rate'] = self._h2h_btts_rate(home_team, away_team, home_history + away_history)

        features['home_goal_diff'] = self._goal_difference(home_history[:10], home_team)
        features['away_goal_diff'] = self._goal_difference(away_history[:10], away_team)

        # BLOCK 4: Poisson Parameters (4 features)
        features['lambda_home'] = features['home_attack_strength'] * features['away_defense_strength'] * (league_avg / 2)
        features['lambda_away'] = features['away_attack_strength'] * features['home_defense_strength'] * (league_avg / 2)
        features['expected_total_goals'] = features['lambda_home'] + features['lambda_away']
        features['poisson_over25'] = self._poisson_over_x(features['lambda_home'], features['lambda_away'], 2.5)

        # BLOCK 5: Advanced Stats (9 features)
        features['home_wins_pct'] = self._win_rate(home_history[:10], home_team)
        features['away_wins_pct'] = self._win_rate(away_history[:10], away_team)
        features['home_over25_rate'] = self._over_rate(home_history[:10], 2.5)
        features['away_over25_rate'] = self._over_rate(away_history[:10], 2.5)
        features['home_btts_rate'] = self._btts_rate(home_history[:10], home_team)
        features['away_btts_rate'] = self._btts_rate(away_history[:10], away_team)
        features['home_clean_sheet_rate'] = self._clean_sheet_rate(home_history[:10], home_team)
        features['away_clean_sheet_rate'] = self._clean_sheet_rate(away_history[:10], away_team)
        features['combined_form'] = (features['home_points_form_5'] + features['away_points_form_3']) / 2

        return features

    def _goals_avg(self, matches, team_name):
        goals = []
        for m in matches:
            if m.get('homeTeam', {}).get('name') == team_name:
                g = m.get('score', {}).get('fullTime', {}).get('home')
                if g is not None:
                    goals.append(g)
            elif m.get('awayTeam', {}).get('name') == team_name:
                g = m.get('score', {}).get('fullTime', {}).get('away')
                if g is not None:
                    goals.append(g)
        return np.mean(goals) if goals else 0.0

    def _goals_conceded_avg(self, matches, team_name):
        conceded = []
        for m in matches:
            if m.get('homeTeam', {}).get('name') == team_name:
                g = m.get('score', {}).get('fullTime', {}).get('away')
                if g is not None:
                    conceded.append(g)
            elif m.get('awayTeam', {}).get('name') == team_name:
                g = m.get('score', {}).get('fullTime', {}).get('home')
                if g is not None:
                    conceded.append(g)
        return np.mean(conceded) if conceded else 0.0

    def _points_form(self, matches, team_name):
        points = 0
        for m in matches:
            home = m.get('homeTeam', {}).get('name')
            away = m.get('awayTeam', {}).get('name')
            hg = m.get('score', {}).get('fullTime', {}).get('home')
            ag = m.get('score', {}).get('fullTime', {}).get('away')

            if hg is None or ag is None:
                continue

            if home == team_name:
                if hg > ag:
                    points += 3
                elif hg == ag:
                    points += 1
            elif away == team_name:
                if ag > hg:
                    points += 3
                elif ag == hg:
                    points += 1
        return points

    def _goal_difference(self, matches, team_name):
        scored, conceded = [], []
        for m in matches:
            if m.get('homeTeam', {}).get('name') == team_name:
                hg = m.get('score', {}).get('fullTime', {}).get('home')
                ag = m.get('score', {}).get('fullTime', {}).get('away')
                if hg is not None and ag is not None:
                    scored.append(hg)
                    conceded.append(ag)
            elif m.get('awayTeam', {}).get('name') == team_name:
                hg = m.get('score', {}).get('fullTime', {}).get('home')
                ag = m.get('score', {}).get('fullTime', {}).get('away')
                if hg is not None and ag is not None:
                    scored.append(ag)
                    conceded.append(hg)
        return sum(scored) - sum(conceded) if scored else 0

    def _h2h_goals_avg(self, home_team, away_team, all_matches):
        h2h = [m for m in all_matches 
               if (m.get('homeTeam', {}).get('name') == home_team and m.get('awayTeam', {}).get('name') == away_team)
               or (m.get('homeTeam', {}).get('name') == away_team and m.get('awayTeam', {}).get('name') == home_team)]

        goals = []
        for m in h2h:
            hg = m.get('score', {}).get('fullTime', {}).get('home')
            ag = m.get('score', {}).get('fullTime', {}).get('away')
            if hg is not None and ag is not None:
                goals.append(hg + ag)
        return np.mean(goals) if goals else 2.75

    def _h2h_btts_rate(self, home_team, away_team, all_matches):
        h2h = [m for m in all_matches 
               if (m.get('homeTeam', {}).get('name') == home_team and m.get('awayTeam', {}).get('name') == away_team)
               or (m.get('homeTeam', {}).get('name') == away_team and m.get('awayTeam', {}).get('name') == home_team)]

        btts_count = 0
        for m in h2h:
            hg = m.get('score', {}).get('fullTime', {}).get('home', 0)
            ag = m.get('score', {}).get('fullTime', {}).get('away', 0)
            if hg > 0 and ag > 0:
                btts_count += 1
        return btts_count / len(h2h) if h2h else 0.5

    def _over_rate(self, matches, threshold):
        count = 0
        for m in matches:
            hg = m.get('score', {}).get('fullTime', {}).get('home', 0)
            ag = m.get('score', {}).get('fullTime', {}).get('away', 0)
            if hg + ag > threshold:
                count += 1
        return count / len(matches) if matches else 0.0

    def _btts_rate(self, matches, team_name):
        count = 0
        for m in matches:
            hg = m.get('score', {}).get('fullTime', {}).get('home', 0)
            ag = m.get('score', {}).get('fullTime', {}).get('away', 0)
            if hg > 0 and ag > 0:
                count += 1
        return count / len(matches) if matches else 0.0

    def _win_rate(self, matches, team_name):
        wins = 0
        for m in matches:
            home = m.get('homeTeam', {}).get('name')
            hg = m.get('score', {}).get('fullTime', {}).get('home')
            ag = m.get('score', {}).get('fullTime', {}).get('away')

            if hg is None or ag is None:
                continue

            if home == team_name and hg > ag:
                wins += 1
            elif home != team_name and ag > hg:
                wins += 1
        return wins / len(matches) if matches else 0.0

    def _clean_sheet_rate(self, matches, team_name):
        cs = 0
        for m in matches:
            home = m.get('homeTeam', {}).get('name')
            hg = m.get('score', {}).get('fullTime', {}).get('home', 0)
            ag = m.get('score', {}).get('fullTime', {}).get('away', 0)

            if home == team_name and ag == 0:
                cs += 1
            elif home != team_name and hg == 0:
                cs += 1
        return cs / len(matches) if matches else 0.0

    def _poisson_over_x(self, lambda_home, lambda_away, threshold):
        from scipy.stats import poisson
        prob_under = 0.0
        for h in range(int(threshold) + 2):
            for a in range(int(threshold) + 2 - h):
                if h + a <= threshold:
                    prob_under += poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
        return 1 - prob_under


if __name__ == "__main__":
    fe = FeatureEngineer()
    print("✓ Feature Engineering ready - 36 features")
