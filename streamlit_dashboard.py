#!/usr/bin/env python3
"""
Streamlit Dashboard - Real-Time UI
"""

import streamlit as st
import pandas as pd
import sqlite3
import time
from datetime import datetime

st.set_page_config(
    page_title="⚽ Over 2.5 + BTTS Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("🎯 Real-Time Betting System - Over 2.5 + BTTS")

# Sidebar
st.sidebar.header("⚙️ Filtry")
min_prob = st.sidebar.slider("Min. prawdopodobieństwo Over 2.5", 0.5, 0.95, 0.65, 0.05)
min_btts = st.sidebar.slider("Min. prawdopodobieństwo BTTS", 0.5, 0.95, 0.60, 0.05)
refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=True)

# Metrics
col1, col2, col3 = st.columns(3)

try:
    conn = sqlite3.connect("data/predictions.db")

    # Active predictions
    active_count = pd.read_sql(
        "SELECT COUNT(*) as cnt FROM predictions WHERE kickoff_utc > datetime('now')", 
        conn
    ).iloc[0]['cnt']

    col1.metric("🎲 Aktywne predykcje", active_count)

    # Accuracy (if results available)
    try:
        accuracy_data = pd.read_sql("""
            SELECT 
                ROUND(AVG(CASE WHEN (r.over25_actual = 1 AND p.over25_prob >= 0.65) OR
                                     (r.over25_actual = 0 AND p.over25_prob < 0.65) THEN 1.0 ELSE 0.0 END) * 100, 1) as acc
            FROM predictions p
            JOIN results r ON p.fixture_id = r.fixture_id
            WHERE p.predicted_at >= datetime('now', '-7 days')
        """, conn)

        if not accuracy_data.empty and accuracy_data.iloc[0]['acc'] is not None:
            col2.metric("📊 Accuracy (7 dni)", f"{accuracy_data.iloc[0]['acc']:.1f}%")
        else:
            col2.metric("📊 Accuracy (7 dni)", "N/A")
    except:
        col2.metric("📊 Accuracy (7 dni)", "N/A")

    col3.metric("⏰ Ostatnia aktualizacja", datetime.now().strftime("%H:%M:%S"))

    # Main table
    st.subheader("📋 Najlepsze zakłady (LIVE)")

    df = pd.read_sql(f"""
        SELECT 
            home_team as "Gospodarz",
            away_team as "Gość",
            strftime('%d.%m %H:%M', kickoff_utc) as "Kickoff",
            ROUND(over25_prob * 100, 1) as "Over 2.5 %",
            ROUND(btts_prob * 100, 1) as "BTTS %",
            over25_confidence as "Confidence",
            ROUND(expected_goals, 2) as "xG",
            league as "Liga"
        FROM predictions
        WHERE over25_prob >= {min_prob}
        AND btts_prob >= {min_btts}
        AND kickoff_utc > datetime('now')
        ORDER BY over25_prob DESC
        LIMIT 50
    """, conn)

    conn.close()

    if df.empty:
        st.warning("Brak predykcji spełniających kryteria. Obniż filtry lub poczekaj na nowe mecze.")
    else:
        # Color coding
        def color_prob(val):
            if val >= 75:
                return 'background-color: #90EE90'
            elif val >= 65:
                return 'background-color: #FFE4B5'
            return ''

        styled_df = df.style.applymap(color_prob, subset=['Over 2.5 %', 'BTTS %'])
        st.dataframe(styled_df, use_container_width=True, height=600)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Pobierz CSV",
            csv,
            "predictions.csv",
            "text/csv",
            key='download-csv'
        )

except Exception as e:
    st.error(f"Błąd połączenia z bazą danych: {e}")
    st.info("Upewnij się, że monitor działa: `python core/real_time_monitor.py`")

# Auto-refresh
if refresh:
    time.sleep(30)
    st.rerun()
