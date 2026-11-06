"""Reproduce county-level Clinton vote analysis with leakage-aware splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COLUMNS = [
    "county", "percentVoting", "age", "savings", "income", "poverty",
    "percentVeterans", "percentFemale", "popDensity", "percentNursing", "crimeIndex",
]
REGRESSORS = COLUMNS[2:]


def load_data(path):
    # County names can contain spaces; delimit fields only at runs of 2+ spaces.
    frame = pd.read_csv(path, sep=r"\s{2,}", header=None, names=COLUMNS, engine="python")
    if frame.empty or frame.isna().any().any():
        raise ValueError("Voting data must be nonempty and complete.")
    numeric = frame.drop(columns="county").apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Voting data must contain finite numeric values.")
    return pd.concat([frame["county"], numeric], axis=1)


def _greedy_forward(X_train, y_train, candidates):
    remaining = list(candidates)
    chosen = []
    scores = []
    for _ in range(len(remaining)):
        best_feature, best_score = None, -np.inf
        for feature in remaining:
            model = LinearRegression().fit(X_train[chosen + [feature]], y_train)
            score = model.score(X_train[chosen + [feature]], y_train)
            if score > best_score:
                best_feature, best_score = feature, score
        chosen.append(best_feature)
        remaining.remove(best_feature)
        scores.append(float(best_score))
    return chosen, scores


def analyze(frame, seed=42):
    X = frame[REGRESSORS]
    target = frame["percentVoting"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.3, random_state=seed
    )
    order, train_r2 = _greedy_forward(X_train, y_train, REGRESSORS)
    selected_count = int(np.argmax(train_r2) + 1)
    selected = order[:selected_count]
    regressor = LinearRegression().fit(X_train[selected], y_train)
    predictions = regressor.predict(X_test[selected])

    labels = (frame["percentVoting"] >= 40.0).astype(int)
    feature_names = ["poverty", "popDensity"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        frame[feature_names], labels, test_size=0.3, random_state=seed, stratify=labels
    )
    if yc_train.nunique() != 2 or yc_test.nunique() != 2:
        raise ValueError("Both classes must be represented in train and test splits.")
    folds = min(5, int(yc_train.value_counts().min()))
    if folds < 2:
        raise ValueError("Not enough minority-class examples for cross-validation.")
    classifier = make_pipeline(StandardScaler(), KNeighborsClassifier())
    search = GridSearchCV(
        classifier,
        {"kneighborsclassifier__n_neighbors": [1, 3, 5, 10, 20, 50, 100]},
        cv=folds,
        scoring="balanced_accuracy",
        n_jobs=1,
        error_score="raise",
    ).fit(Xc_train, yc_train)
    class_predictions = search.predict(Xc_test)
    return {
        "rows": len(frame),
        "regression": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "forward_selection_order": order,
            "selected_features_by_training_r2": selected,
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        },
        "classification": {
            "target_definition": "percentVoting >= 40.0",
            "train_rows": len(Xc_train),
            "test_rows": len(Xc_test),
            "best_neighbors": int(search.best_params_["kneighborsclassifier__n_neighbors"]),
            "training_cv_balanced_accuracy": float(search.best_score_),
            "test_accuracy": float(accuracy_score(yc_test, class_predictions)),
            "test_balanced_accuracy": float(balanced_accuracy_score(yc_test, class_predictions)),
            "test_prevalence": float(yc_test.mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=Path("clinton1.dat"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = analyze(load_data(args.data), args.seed)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
