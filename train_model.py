
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

# Start small; increase after the pipeline works.
SAMPLE_PER_FILE = 20_000
SAMPLE_PER_CHUNK = 2_000
CHUNK_SIZE = 100_000
RANDOM_STATE = 42

# Columns that can identify a particular flow rather than
# generalizable network behavior.
EXCLUDED_COLUMNS = {
    "flow id",
    "source ip",
    "destination ip",
    "src ip",
    "dst ip",
    "timestamp",
    "simillarhttp",
}


def normalize_column(name):
    """Normalize column names for consistent comparisons."""
    name = str(name).replace("\ufeff", "").strip()
    return re.sub(r"\s+", " ", name)


def load_dataset():
    files = sorted(DATA_DIR.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No CSV files found inside the data/ folder."
        )

    sampled_frames = []

    for file in files:
        print(f"Reading: {file.name}")
        collected = 0
        file_parts = []

        try:
            reader = pd.read_csv(
                file,
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )

            for chunk in reader:
                chunk.columns = [
                    normalize_column(c) for c in chunk.columns
                ]

                label_col = next(
                    (
                        c for c in chunk.columns
                        if c.lower() == "label"
                    ),
                    None,
                )

                if label_col is None:
                    raise ValueError(
                        f"No Label column found in {file.name}"
                    )

                chunk = chunk.rename(columns={label_col: "Label"})

                # Sample a portion of each chunk to limit memory.
                remaining = SAMPLE_PER_FILE - collected

                if remaining <= 0:
                    break

                take = min(SAMPLE_PER_CHUNK, remaining, len(chunk))

                if take == 0:
                    continue

                part = chunk.sample(
                    n=take,
                    random_state=RANDOM_STATE,
                )
                file_parts.append(part)
                collected += len(part)

            if file_parts:
                sampled = pd.concat(file_parts, ignore_index=True)
                sampled_frames.append(sampled)
                print(f"  Sampled {len(sampled):,} rows")

        except Exception as exc:
            raise RuntimeError(
                f"Could not process {file}: {exc}"
            ) from exc

    if not sampled_frames:
        raise ValueError("No usable rows were loaded.")

    data = pd.concat(sampled_frames, ignore_index=True)

    # Clean label values such as "BENIGN " and "DDoS".
    data["Label"] = data["Label"].astype(str).str.strip()
    data = data[data["Label"].ne("")]
    data = data[data["Label"].str.lower().ne("nan")]

    # Binary classification: BENIGN versus every attack category.
    data["target"] = (
        data["Label"].str.upper() != "BENIGN"
    ).astype(int)

    # Exclude labels and flow-specific identifiers.
    excluded = EXCLUDED_COLUMNS | {"label", "target"}

    feature_candidates = [
        col for col in data.columns
        if col.lower() not in excluded
    ]

    X = data[feature_candidates].copy()

    # Keep numeric features only.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that contain no usable numeric data.
    X = X.dropna(axis=1, how="all")

    # A feature must have at least two distinct observed values.
    X = X.loc[:, X.nunique(dropna=True) > 1]

    # Preserve the original data's label as a readable class.
    y = data.loc[X.index, "target"]

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable numeric features remain.")

    if y.nunique() < 2:
        raise ValueError(
            "Only one target class was loaded. Include CSV files "
            "containing both BENIGN and attack traffic."
        )

    # Remove exact duplicate feature rows to reduce repetition.
    # Note: identical feature rows can legitimately occur in traffic.
    # This is a convenience, not a substitute for group/time splitting.
    combined = X.copy()
    combined["__target__"] = y
    combined = combined.drop_duplicates()
    y = combined.pop("__target__")
    X = combined

    print("\nDataset summary")
    print("Rows:", len(X))
    print("Features:", len(X.columns))
    print("Class counts:")
    print(y.map({0: "BENIGN", 1: "ATTACK"}).value_counts())

    return X, y


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()

    # Stratification helps preserve the class ratio in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0,
    )

    print("\nEvaluation results")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_attack": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall_attack": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_attack": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": report,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": list(X.columns),
    }

    joblib.dump(
        {
            "model": model,
            "features": list(X.columns),
            "classes": {0: "BENIGN", 1: "ATTACK"},
        },
        MODEL_DIR / "traffic_model.joblib",
    )

    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel saved to models/traffic_model.joblib")
    print("Metrics saved to reports/metrics.json")


if __name__ == "__main__":
    main()
