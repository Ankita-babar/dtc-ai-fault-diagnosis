"""
Diagnosis engine v2 - answers for ANY code a car can send.

Order of evidence (most to least reliable)
  1. expert knowledge base (158 gold codes)               -> exact lookup
  2. code found in the SAE generic table (2,066 codes)    -> description known, severity/system by ML
  3. description supplied by the user/scan tool           -> ML on the text
  4. code only (manufacturer-specific / unlisted)         -> ML on the code pattern (P/B/C/U, range) - LOW confidence
Solutions for 2-4 are retrieved from the most similar expert-labelled code (shown as 'similar_to').

Usage
  python src/diagnose.py P0301 P0106
  python src/diagnose.py P1A2B                          # unknown code, no description
  python src/diagnose.py "P1A2B:Fuel rail pressure sensor circuit low"
"""
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from dtc_rules import LETTER, range_group

ROOT = Path(__file__).resolve().parents[1]
ALERT = {"High": "Critical", "Medium": "Warning", "Low": "Warning"}
DEDUCT = {"High": 30, "Medium": 15, "Low": 5}
RANK = {"High": 0, "Medium": 1, "Low": 2}
VERIFY_BELOW = 0.60        # severity confidence under this -> flag "verify with a technician"
GENERIC = {"circuit", "control", "open", "low", "high", "range", "performance", "input", "output", "signal", "intermittent",
           "erratic", "malfunction", "too", "bank", "sensor", "switch", "system", "the", "of", "with", "from", "and", "or", "a", "b", "c", "d"}
SIM_MIN = 0.50          # below this similarity we do NOT borrow a neighbour's solution


def clean(t):
    import re
    return " ".join(w for w in re.findall(r"[a-z]+", str(t).lower()) if w not in GENERIC and len(w) > 1)


VALID = r"^[PBCU][0-9A-F]{4}$"


class DiagnosisEngine:
    def __init__(self):
        self.kb = pd.read_csv(ROOT / "data" / "shortlisted_dtc_project.csv").set_index("code")
        sae = pd.read_csv(ROOT / "data" / "sae_dtc_table.csv")
        self.sae = dict(zip(sae.code, sae.description))
        m = ROOT / "models"
        self.sev, self.grp = joblib.load(m / "severity_model.joblib"), joblib.load(m / "group_model.joblib")
        self.sev_c, self.grp_c = joblib.load(m / "severity_code_only_model.joblib"), joblib.load(m / "group_code_only_model.joblib")
        # retrieval index over the expert codes (for solutions)
        self.kb_codes = list(self.kb.index)
        self.tfidf = TfidfVectorizer(sublinear_tf=True)
        self.kb_vec = self.tfidf.fit_transform(self.kb.description.map(clean))

    @staticmethod
    def _frame(code, desc):
        return pd.DataFrame({"code": [code], "description": [desc or ""], "letter": [code[:1]], "p1": [code[:2]], "p2": [code[:3]]})

    def _predict(self, code, desc):
        x = self._frame(code, desc)
        sev_m, grp_m = (self.sev, self.grp) if desc else (self.sev_c, self.grp_c)
        sp, gp = sev_m.predict_proba(x)[0], grp_m.predict_proba(x)[0]
        return (sev_m.classes_[sp.argmax()], float(sp.max()), grp_m.classes_[gp.argmax()], float(gp.max()))

    def _solution(self, desc, group):
        """Most similar expert code INSIDE the predicted system -> its first_check (never borrow from another system)."""
        sims = (self.kb_vec @ self.tfidf.transform([clean(desc)]).T).toarray().ravel()
        same = (self.kb.maintenance_group.values == group)
        pool = np.where(same, sims, -1.0) if same.any() else sims
        i = int(pool.argmax())
        if sims[i] < SIM_MIN:
            return (f"Inspect the {group} system - check the related sensor/actuator, wiring and connectors; "
                    f"confirm with a technician", None, float(sims[i]))
        return self.kb.first_check.iloc[i], self.kb_codes[i], float(sims[i])

    def lookup(self, code, description=None):
        code = str(code).strip().upper()
        if code in self.kb.index:                                    # 1. expert KB
            r = self.kb.loc[code]
            return {"code": code, "source": "knowledge base", "evidence": "Expert-labelled knowledge base",
                    "description": r.description, "severity": r.severity, "alert_level": r.alert_level,
                    "health_deduction": int(r.health_deduction), "maintenance_group": r.maintenance_group,
                    "first_check": r.first_check, "monitor_pid": None if pd.isna(r.related_live_pid) else r.related_live_pid,
                    "readable": r.readable_on_real_car.split(" - ")[0], "needs_verification": False}
        if not pd.Series([code]).str.match(VALID)[0]:
            return {"code": code, "source": "invalid code", "evidence": "Not a valid DTC format", "description": "Invalid code format",
                    "severity": "Unknown", "alert_level": "Unknown", "health_deduction": 0, "maintenance_group": None,
                    "first_check": "Re-read the code - a DTC looks like P0301", "monitor_pid": None, "needs_verification": True}
        desc, from_table = (description or self.sae.get(code)), False
        if not description and code in self.sae: from_table = True
        sev, sc, grp, gc = self._predict(code, desc)
        out = {"code": code, "monitor_pid": None, "severity_confidence": round(sc, 2), "group_confidence": round(gc, 2)}
        if desc:                                                      # 2/3. description available
            sol, sim_code, sim = self._solution(desc, grp)
            out.update(source="ML prediction", description=desc, first_check=sol, similar_to=sim_code, similarity=round(sim, 2),
                       evidence="SAE generic code table + ML" if from_table else "Description provided + ML")
        else:                                                         # 4. code pattern only
            cat = LETTER.get(code[0], "Unknown")
            sev = sev if RANK[sev] <= 1 else "Medium"                 # safety floor: never call an unexplained code 'Low'
            grp = grp or range_group(code)
            out.update(source="code-pattern estimate", evidence="Code pattern only - low confidence",
                       description=f"Unlisted {cat} code (manufacturer-specific or not in tables)",
                       first_check=f"Look up {code} in the manufacturer's service data, read freeze-frame data, and inspect the {grp} system",
                       similar_to=None, similarity=None)
        out.update(severity=sev, alert_level=ALERT[sev], health_deduction=DEDUCT[sev], maintenance_group=grp,
                   needs_verification=bool(sc < VERIFY_BELOW or not desc))
        return out

    def diagnose(self, entries):
        faults = [self.lookup(*(e if isinstance(e, tuple) else (e, None))) for e in entries]
        faults.sort(key=lambda f: RANK.get(f["severity"], 3))
        score = max(0, 100 - sum(f["health_deduction"] for f in faults))
        status = "CRITICAL" if any(f["alert_level"] == "Critical" for f in faults) else \
                 "WARNING" if any(f["alert_level"] == "Warning" for f in faults) else "NORMAL"
        return {"status": status, "health_score": score, "n_faults": len(faults), "faults": faults}


def print_report(rep):
    print(f"\nVEHICLE STATUS : {rep['status']}   |   HEALTH SCORE : {rep['health_score']}/100   |   FAULTS : {rep['n_faults']}")
    print("=" * 80)
    for f in rep["faults"]:
        flag = "  [VERIFY WITH TECHNICIAN]" if f.get("needs_verification") else ""
        print(f"{f['code']}  [{f['alert_level']} / {f['severity']}]  {f['evidence']}{flag}")
        print(f"  Fault     : {f['description']}")
        print(f"  System    : {f['maintenance_group']}")
        print(f"  First step: {f['first_check']}")
        if f.get("similar_to"): print(f"  Based on  : similar expert code {f['similar_to']} (similarity {f['similarity']})")
        if "severity_confidence" in f: print(f"  Confidence: severity {f['severity_confidence']:.0%}, system {f['group_confidence']:.0%}")
        print("-" * 80)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("codes", nargs="+"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    entries = [tuple(c.split(":", 1)) if ":" in c else c for c in a.codes]
    rep = DiagnosisEngine().diagnose(entries)
    print(json.dumps(rep, indent=2, default=str) if a.json else "", end="")
    if not a.json: print_report(rep)


if __name__ == "__main__":
    main()
