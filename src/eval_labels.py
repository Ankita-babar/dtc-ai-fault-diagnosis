"""
Independent test: label reports/label_me_100.csv by hand (columns your_severity = Low/Medium/High,
your_group = one of the 24 group names in data/shortlisted_dtc_project.csv), then run:
    python src/eval_labels.py
It reports how often the engine agrees with YOUR labels (codes the model never trained on as gold).
"""
from pathlib import Path
import pandas as pd
from diagnose import DiagnosisEngine

f = Path(__file__).resolve().parents[1] / "reports" / "label_me_100.csv"
d = pd.read_csv(f).dropna(subset=["your_severity"])
if d.empty: raise SystemExit("Fill the your_severity / your_group columns first.")
e = DiagnosisEngine(); r = [e.lookup(c) for c in d.code]
sev_ok = (pd.Series([x["severity"] for x in r], index=d.index) == d.your_severity.str.strip().str.title())
print(f"Labelled codes: {len(d)} | severity agreement: {sev_ok.mean():.1%}")
if d.your_group.notna().any():
    g = d.dropna(subset=["your_group"]); gp = pd.Series([x["maintenance_group"] for x in r], index=d.index)[g.index]
    print(f"System agreement: {(gp == g.your_group.str.strip()).mean():.1%} on {len(g)} codes")
