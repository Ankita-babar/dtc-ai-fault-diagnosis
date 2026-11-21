"""
Train + evaluate the DTC models (v2: expanded corpus).

Evidence rules
  * TEST SET = the 158 expert (gold) codes only, grouped 5-fold CV (twin codes such as Bank 1/Bank 2 stay together).
  * Silver codes (1,940 SAE codes labelled by src/dtc_rules.py) are used ONLY as extra training data.
    Codes that are in the gold set are never used as silver. Group->severity prior is recomputed from
    training-fold gold only.
Part 1  model comparison on gold-only data (repeated grouped 5-fold CV, mean +- std)         -> reports/model_comparison.csv
Part 2  does extra silver data / code structure help? -> reports/expansion_experiment.csv
Part 3  fit final models on gold + silver and save   -> models/*.joblib
Run:  python src/train.py
"""
import json, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from dtc_rules import group_prior, rule_group, rule_severity

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
SILVER_W = 0.5            # weight of a silver (rule-labelled) sample relative to a gold sample


def add_struct(df):
    df = df.copy()
    df["letter"], df["p1"], df["p2"] = df.code.str[:1], df.code.str[:2], df.code.str[:3]
    return df


def make_pipeline(clf, text=True, struct=False):
    parts = []
    if text:
        parts += [("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), "description"),
                  ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True), "description")]
    if struct:
        parts += [("code", OneHotEncoder(handle_unknown="ignore"), ["letter", "p1", "p2"])]
    return Pipeline([("features", ColumnTransformer(parts)), ("clf", clf)])


def LR(): return LogisticRegression(max_iter=4000, C=10, class_weight="balanced")


def family_groups(desc):
    g = desc.str.lower().str.replace(r"\d+", "", regex=True)
    return g.str.replace(r"\b[a-z]\b", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def silver_for(sae, gold_codes, gold_train):
    """Silver rows = SAE codes NOT in gold, labelled by rules with a prior from training-fold gold."""
    s = sae[~sae.code.isin(gold_codes)].copy()
    prior = group_prior(gold_train)
    s["maintenance_group"] = [rule_group(c, d) for c, d in zip(s.code, s.description)]
    s["severity"] = [rule_severity(d, g, prior) for d, g in zip(s.description, s.maintenance_group)]
    return add_struct(s)


def scores(y, pred):
    return {"accuracy": accuracy_score(y, pred),
            "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0)}


REPEATS = 10         # repeated grouped 5-fold CV (different random group partitions) -> mean +- std


def grouped_folds(groups, k=5, seed=0):
    """Version-independent grouped k-fold: whole 'families' are shuffled and placed in the currently smallest fold."""
    rng = np.random.RandomState(seed)
    sizes = groups.value_counts()
    order = list(sizes.index); rng.shuffle(order)
    fold_of, load = {}, np.zeros(k)
    for g in order:
        f = int(np.argmin(load + rng.rand(k) * 1e-6)); fold_of[g] = f; load[f] += sizes[g]
    fid = groups.map(fold_of).values
    return [(np.where(fid != f)[0], np.where(fid == f)[0]) for f in range(k)]


def cv_run(gold, sae, target, fit_predict, repeats=REPEATS):
    """Repeated grouped 5-fold CV on gold only. Returns mean of each metric and std of accuracy / F1."""
    groups, res = family_groups(gold.description), []
    for seed in range(repeats):
        pred = pd.Series(index=gold.index, dtype=object)
        for tr, te in grouped_folds(groups, 5, seed):
            trg, teg = gold.iloc[tr], gold.iloc[te]
            pred.iloc[te] = fit_predict(trg, silver_for(sae, set(gold.code), trg), teg)
        res.append(scores(gold[target], pred))
    df = pd.DataFrame(res); out = df.mean().to_dict()
    out["accuracy_std"], out["macro_f1_std"] = df.accuracy.std(), df.macro_f1.std()
    return out


def main():
    import sys
    final_only = "--final-only" in sys.argv
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    gold = add_struct(pd.read_csv(ROOT / "data" / "shortlisted_dtc_project.csv"))
    sae = pd.read_csv(ROOT / "data" / "sae_dtc_table.csv")
    print(f"gold: {len(gold)} | silver available: {(~sae.code.isin(gold.code)).sum()}\n")

    if not final_only:
        # ---------------- Part 1: classifiers on gold-only text ----------------
        zoo = {"Majority baseline": lambda: DummyClassifier(strategy="most_frequent"), "Logistic Regression": LR,
               "Linear SVM": lambda: LinearSVC(C=1, class_weight="balanced"),
               "Random Forest": lambda: RandomForestClassifier(300, class_weight="balanced", random_state=42)}
        rows = []
        for target in ["severity", "maintenance_group"]:
            for name, mk in zoo.items():
                r = cv_run(gold, sae, target, lambda trg, sil, teg, mk=mk, target=target:
                           make_pipeline(mk()).fit(trg, trg[target]).predict(teg))
                rows.append({"task": target, "model": name, **r})
        comp = pd.DataFrame(rows).round(3); comp.to_csv(REPORTS / "model_comparison.csv", index=False)
        print("PART 1 - gold-only, grouped CV\n", comp.to_string(index=False), "\n")

        # ---------------- Part 2: expansion experiment ----------------
        def with_silver(target, text, struct, w):
            def f(trg, sil, teg):
                X = pd.concat([trg, sil]); y = X[target]
                sw = np.r_[np.ones(len(trg)), np.full(len(sil), w)]
                return make_pipeline(LR(), text, struct).fit(X, y, clf__sample_weight=sw).predict(teg)
            return f

        def rules_only(trg, sil, teg):          # no ML at all
            prior = group_prior(trg)
            return [rule_severity(d, rule_group(c, d), prior) for c, d in zip(teg.code, teg.description)]

        exp = []
        for target in ["severity", "maintenance_group"]:
            systems = {
                "A  LR gold-only, text (previous)": lambda trg, sil, teg, t=target: make_pipeline(LR()).fit(trg, trg[t]).predict(teg),
                "B  LR gold+silver, text": with_silver(target, True, False, SILVER_W),
                "C  LR gold+silver, text+code": with_silver(target, True, True, SILVER_W),
                "D  LR gold+silver, code only (no description)": with_silver(target, False, True, SILVER_W),
                "E  LR gold-only, code only": lambda trg, sil, teg, t=target: make_pipeline(LR(), False, True).fit(trg, trg[t]).predict(teg),
            }
            if target == "severity": systems["F  rules only (no ML)"] = rules_only
            else: systems["F  rules only (no ML)"] = lambda trg, sil, teg: [rule_group(c, d) for c, d in zip(teg.code, teg.description)]
            for name, fn in systems.items():
                exp.append({"task": target, "system": name, **cv_run(gold, sae, target, fn)})
        ex = pd.DataFrame(exp).round(3); ex.to_csv(REPORTS / "expansion_experiment.csv", index=False)
        print("PART 2 - expansion experiment (test = gold only)\n", ex.to_string(index=False), "\n")

    # ---------------- Part 3: final models (all gold + all silver) ----------------
    silver = silver_for(sae, set(gold.code), gold)
    full = pd.concat([gold, silver]); sw = np.r_[np.ones(len(gold)), np.full(len(silver), SILVER_W)]
    for target, key in [("severity", "severity"), ("maintenance_group", "group")]:
        joblib.dump(make_pipeline(LR(), True, True).fit(full, full[target], clf__sample_weight=sw), MODELS / f"{key}_model.joblib")
        joblib.dump(make_pipeline(LR(), False, True).fit(full, full[target], clf__sample_weight=sw), MODELS / f"{key}_code_only_model.joblib")
    if not final_only: (REPORTS / "metrics.json").write_text(json.dumps({"n_gold": len(gold), "n_silver": len(silver), "silver_weight": SILVER_W,
        "comparison": comp.to_dict("records"), "expansion": ex.to_dict("records")}, indent=1))
    print("Saved 4 models ->", MODELS)


if __name__ == "__main__":
    main()
