from flask import Flask, jsonify, request
from datetime import datetime
import sqlite3
import json
import os
from noshow_iq.model import predict

app = Flask(__name__)

DB_PATH = os.environ.get("DB_PATH", "/tmp/noshow.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_json TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            probability REAL NOT NULL,
            recommendation TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            training_size INTEGER,
            precision_class_0 REAL,
            recall_class_0 REAL,
            f1_class_0 REAL,
            precision_class_1 REAL,
            recall_class_1 REAL,
            f1_class_1 REAL,
            imbalance_technique TEXT
        )
    """)
    conn.commit()
    return conn


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def make_prediction():
    data = request.get_json()
    result = predict(data)
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO predictions "
            "(timestamp, input_json, risk_level, probability, recommendation) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(),
                json.dumps(data),
                result["risk_level"],
                result["probability"],
                result["recommendation"],
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"SQLite error: {e}")
    return jsonify(result)


@app.route("/history", methods=["GET"])
def history():
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT id, timestamp, input_json, risk_level, "
            "probability, recommendation "
            "FROM predictions ORDER BY timestamp DESC LIMIT 20"
        ).fetchall()
        conn.close()
        docs = []
        for r in rows:
            docs.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "input": json.loads(r["input_json"]),
                "risk_level": r["risk_level"],
                "probability": r["probability"],
                "recommendation": r["recommendation"],
            })
        return jsonify(docs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    try:
        conn = get_db()
        row = conn.execute("""
            SELECT
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END)
                    AS high_risk_count,
                SUM(CASE WHEN risk_level = 'low' THEN 1 ELSE 0 END)
                    AS low_risk_count,
                AVG(probability) AS average_probability
            FROM predictions
        """).fetchone()
        last_run = conn.execute(
            "SELECT timestamp FROM training_runs "
            "ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
        stats_data = {
            "total_predictions": row["total_predictions"] or 0,
            "high_risk_count": row["high_risk_count"] or 0,
            "low_risk_count": row["low_risk_count"] or 0,
            "average_probability": row["average_probability"],
            "last_trained": last_run["timestamp"] if last_run else None,
        }
        return jsonify(stats_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
