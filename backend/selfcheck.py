import http.client
import json
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
proc = subprocess.Popen([sys.executable, str(BASE_DIR / "main.py")], cwd=BASE_DIR)
time.sleep(3)

try:
    conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=10)

    conn.request("GET", "/health")
    r = conn.getresponse()
    data = json.loads(r.read())
    assert r.status == 200, f"Health expected 200, got {r.status}"
    assert data["engine"] == "sklearn", f"Wrong engine: {data}"

    conn.request("POST", "/analyze",
                 json.dumps({"text": "This is amazing!"}).encode(),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    data = json.loads(r.read())
    score = data["results"][0]["compound"]
    assert score > 0, f"Expected positive score, got {score}"

    conn.request("POST", "/analyze",
                 json.dumps({"text": "This is terrible."}).encode(),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    data = json.loads(r.read())
    score = data["results"][0]["compound"]
    assert score < 0, f"Expected negative score, got {score}"

    conn.request("GET", "/metrics")
    r = conn.getresponse()
    data = json.loads(r.read())
    assert data["total_analyzed"] >= 2, f"Expected >=2, got {data['total_analyzed']}"

    print("All checks passed.")
    sys.exit(0)
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
finally:
    proc.terminate()
    proc.wait()
