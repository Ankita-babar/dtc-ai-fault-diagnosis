"""Sanity tests:  python src/test_engine.py"""
from diagnose import DiagnosisEngine
from read_obd import read_codes

e = DiagnosisEngine()
r = e.diagnose(["P0301"]); assert r["status"] == "CRITICAL" and r["health_score"] == 70
r = e.diagnose(["P0128"]); assert r["status"] == "WARNING" and r["health_score"] == 95
assert e.diagnose([])["status"] == "NORMAL"
f = e.lookup("P0001"); assert f["source"] == "ML prediction" and "Fuel Volume" in f["description"] and f["first_check"]   # SAE-table code, not in expert list
f = e.lookup("P1A2B", "Fuel rail pressure sensor circuit low"); assert f["source"] == "ML prediction" and f["severity"] == "High"
f = e.lookup("P1A2B"); assert f["source"] == "code-pattern estimate" and f["needs_verification"] and f["severity"] != "Low"
assert e.lookup("P03")["severity"] == "Unknown"

class R:                                   # fake python-OBD response / connection (no car needed)
    def __init__(s, v): s.value = v
    def is_null(s): return not s.value
class Conn:
    def query(s, cmd): return R([("P0301", "x"), ("P1A2B", "")] if cmd == "stored" else [("P0420", "y")])
assert read_codes(Conn(), [("stored", "stored"), ("pending", "pending")]) == {"P0301": "stored", "P1A2B": "stored", "P0420": "pending"}
print("All tests passed")
