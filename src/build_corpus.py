"""
Build data/dtc_corpus.csv = 158 expert-labelled 'gold' codes + every other SAE generic code
(from data/sae_dtc_table.csv) with 'silver' labels from src/dtc_rules.py.
Also writes reports/label_me_100.csv - a random sample for the team to label by hand (true independent test).
Run:  python src/build_corpus.py
"""
from pathlib import Path
import pandas as pd
from dtc_rules import group_prior, rule_group, rule_severity

ROOT = Path(__file__).resolve().parents[1]


def build():
    gold = pd.read_csv(ROOT / "data" / "shortlisted_dtc_project.csv")
    sae = pd.read_csv(ROOT / "data" / "sae_dtc_table.csv")
    prior = group_prior(gold)
    sil = sae[~sae.code.isin(gold.code)].copy()
    sil["maintenance_group"] = [rule_group(c, d) for c, d in zip(sil.code, sil.description)]
    sil["severity"] = [rule_severity(d, g, prior) for d, g in zip(sil.description, sil.maintenance_group)]
    sil["label_source"], sil["first_check"] = "silver (rule)", ""
    g = gold[["code", "description", "maintenance_group", "severity", "first_check"]].assign(label_source="gold (expert)")
    corpus = pd.concat([g, sil], ignore_index=True)
    corpus["letter"] = corpus.code.str[0]
    return gold, corpus


if __name__ == "__main__":
    gold, corpus = build()
    corpus.to_csv(ROOT / "data" / "dtc_corpus.csv", index=False)
    (ROOT / "reports").mkdir(exist_ok=True)
    s = corpus[corpus.label_source != "gold (expert)"].sample(100, random_state=7)[["code", "description"]]
    s.assign(your_severity="", your_group="").to_csv(ROOT / "reports" / "label_me_100.csv", index=False)
    print(corpus.label_source.value_counts().to_string())
    print(corpus.groupby("label_source").severity.value_counts().unstack().to_string())
    print("letters:", corpus.letter.value_counts().to_dict(), "| groups:", corpus.maintenance_group.nunique())
