# AI-Powered DTC Diagnosis – answers for ANY code a car can send

## How an unknown code is handled (most reliable -> least)
| Level | When | Source of answer |
|---|---|---|
| 1 | code in the 158 expert codes | exact lookup |
| 2 | code in the SAE generic table (2,066 codes) | description known -> ML predicts severity + system |
| 3 | you/the scan tool supply a description | ML on the text |
| 4 | manufacturer-specific / unlisted code | ML on the code pattern (P/B/C/U + range) -> flagged **LOW CONFIDENCE, verify** |

Solutions for levels 2-4 come from the most similar expert code *in the same system* (shown as "Based on");
if nothing similar enough exists, a generic "inspect the <system>" step is given instead of a wrong guess.
Unexplained codes are never called "Low" (safety floor = Medium).
**Limit:** the real meaning of a manufacturer-specific code (e.g. P1A2B) exists only in the maker's service data; no model can know it.

## Data
* `data/shortlisted_dtc_project.csv` – 158 expert (gold) codes = the TEST set.
* `data/sae_dtc_table.csv` – 2,066 SAE generic codes + descriptions (from python-OBD, GPLv2 – cite it).
* `data/dtc_corpus.csv` – both merged; 1,940 extra codes carry *silver* labels from `src/dtc_rules.py`.

## Run
```
python -m pip install -r requirements.txt
python src/build_corpus.py     # rebuild corpus
python src/train.py            # evaluate + train + save models/
python src/diagnose.py P0301 P0001 P1A2B
python src/test_engine.py
python src/server.py           # dashboard on http://127.0.0.1:8000
```

## Real car (ELM327 adapter in the OBD-II port, ignition ON)
```
python src/read_obd.py --port /dev/tty.usbserial-XXXX --send     # then press "Load codes read from car" in the dashboard
python src/read_obd.py --simulate P0301 P1A2B                    # no car: demo
```
Reads stored (Mode 03) + pending (Mode 07) codes; never clears codes. *Not yet tested on a real vehicle.*

## Honest evaluation (test = the 158 gold codes only; mean of 10 repeated grouped 5-fold CV runs, std of accuracy ~3 points)
See `reports/model_comparison.csv` and `reports/expansion_experiment.csv`.
Clean numbers (no rules involved), accuracy: severity - baseline 48.5 %, Logistic Regression 81.2 %, Linear SVM 82.0 %, Random Forest 70.4 %;
system (24 classes) - baseline 10.8 %, LR 76.6 %, SVM 77.6 %, RF 70.5 %. Code pattern only (no description): severity 68.7 %, system 53.3 %.
Training with the extra silver codes changes severity by about +1 point (within noise); system accuracy rises to ~93 %, but this is
**optimistic** because the rules were written while looking at the gold labels. The real test is independent: label
`reports/label_me_100.csv` by hand, then run `python src/eval_labels.py`.
Note: fold assignment is done by our own `grouped_folds()` so numbers do not change with the scikit-learn version.
