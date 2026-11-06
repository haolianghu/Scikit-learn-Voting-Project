# County-level 1992 presidential vote analysis

The notebook explores county attributes and Clinton's 1992 vote share using
`clinton1.dat`. This file contains historical county-level observations; no
Excel data or other election years are included.

## Reproducible analysis

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python voting_baseline.py
python -m unittest discover -s tests -v
```

The CLI validates the whitespace-delimited input, performs forward feature
selection using training data only, and reports held-out regression metrics.
It also tunes a standardized KNN classifier on the training partition with
stratified cross-validation before a single test evaluation. The binary target
is defined as county vote share at least 40%; candidate neighbor counts are
capped at feasible values. JSON output records feature order, selected
predictors, and test metrics. Use `--seed` to select a different split.

## Interpretation

These are county-level associations, not individual voter behavior. Inferring
individual preferences from aggregate county features would be an ecological
fallacy. The one-year, historical dataset and single holdout do not establish
generalization to other elections, and feature selection uncertainty remains.
Do not use these analyses to make claims about individuals or current voting.
