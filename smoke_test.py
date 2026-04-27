import sys
import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"

def test_health():
    r = requests.get(f"{url}/health")
    assert r.status_code == 200
    print("PASS /health")

def test_predict():
    payload = {
        "gender": 0,
        "age": 30,
        "scholarship": 0,
        "hipertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handcap": 0,
        "sms_received": 1,
        "days_in_advance": 5,
        "appt_day_of_week": 2
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200
    print("PASS /predict")

def test_stats():
    r = requests.get(f"{url}/stats")
    assert r.status_code == 200
    print("PASS /stats")

test_health()
test_predict()
test_stats()
