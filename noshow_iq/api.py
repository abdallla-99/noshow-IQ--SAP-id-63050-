from flask import Flask, jsonify, request
from pymongo import MongoClient
from datetime import datetime
import os
from noshow_iq.model import predict

app = Flask(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["noshow_iq"]
predictions_col = db["predictions"]
training_runs_col = db["training_runs"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def make_prediction():
    data = request.get_json()
    result = predict(data)
    doc = {
        "timestamp": datetime.utcnow(),
        "input": data,
        "risk_level": result["risk_level"],
        "probability": result["probability"],
        "recommendation": result["recommendation"],
    }
    predictions_col.insert_one(doc)
    return jsonify(result)


@app.route("/history", methods=["GET"])
def history():
    docs = list(predictions_col.find().sort("timestamp", -1).limit(20))
    for d in docs:
        d["_id"] = str(d["_id"])
        d["timestamp"] = str(d["timestamp"])
    return jsonify(docs)


@app.route("/stats", methods=["GET"])
def stats():
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "high_risk_count": {
                    "$sum": {"$cond": [{"$eq": ["$risk_level", "high"]}, 1, 0]}
                },
                "low_risk_count": {
                    "$sum": {"$cond": [{"$eq": ["$risk_level", "low"]}, 1, 0]}
                },
                "average_probability": {"$avg": "$probability"},
            }
        }
    ]
    result = list(predictions_col.aggregate(pipeline))
    last_run = training_runs_col.find_one(sort=[("timestamp", -1)])
    stats_data = result[0] if result else {}
    stats_data.pop("_id", None)
    stats_data["last_trained"] = str(last_run["timestamp"]) if last_run else None
    return jsonify(stats_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
