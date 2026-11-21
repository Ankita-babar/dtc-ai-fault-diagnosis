"""
Web server for the dashboard.  Run:  python src/server.py   then open http://127.0.0.1:8000

  GET  /               -> dashboard.html
  GET  /api/codes      -> all known DTCs (for autocomplete)
  GET  /api/metrics    -> model accuracy numbers (reports/metrics.json)
  POST /api/diagnose   -> {"entries":[{"code":"P0301"},{"code":"P1A2B","description":"..."}]}
"""
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from diagnose import DiagnosisEngine

ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__)
engine = DiagnosisEngine()
latest = {"codes": [], "time": None}      # last codes pushed by read_obd.py --send


@app.get("/")
def index():
    return send_from_directory(ROOT, "dashboard.html")


@app.after_request
def allow_file_page(resp):          # lets dashboard.html opened by double-click call this API
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.get("/api/codes")
def codes():
    kb = engine.kb.reset_index()[["code", "description", "severity"]]
    return jsonify(kb.to_dict(orient="records"))


@app.get("/api/metrics")
def metrics():
    return jsonify(json.loads((ROOT / "reports" / "metrics.json").read_text()))


@app.post("/api/ingest")
def ingest():
    from datetime import datetime
    body = request.get_json(silent=True) or {}
    latest["codes"] = [str(c).strip().upper() for c in body.get("codes", [])][:50]
    latest["time"] = datetime.now().strftime("%H:%M:%S")
    return jsonify(latest)


@app.get("/api/latest")
def get_latest():
    return jsonify(latest)


@app.post("/api/diagnose")
def diagnose():
    body = request.get_json(silent=True) or {}
    entries = []
    for e in body.get("entries", [])[:30]:
        code = str(e.get("code", "")).strip().upper()
        if code:
            entries.append((code, (e.get("description") or "").strip() or None))
    return jsonify(engine.diagnose(entries))


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))   # 5000 is blocked on Mac by AirPlay Receiver
    print(f"\nDashboard running -> open http://127.0.0.1:{port}  (Ctrl+C to stop)\n")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False)
