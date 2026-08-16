"""
Closes the loop between data collection and model training.

Every set logged with a known intended_exercise (the label the user declared
before generating/recording it) has its raw sensor readings stored in the
database. This module pulls all of those, re-derives features through the
exact same pipeline used at inference time (predict_model.extract_features),
trains a fresh RandomForest on the accumulated data, evaluates it against
the currently-active model on a held-out split, and — if it trained
successfully — promotes it to be the live model for the rest of the running
process (and persists it to disk so it survives a restart too).

Honesty note: this is a small-scale, illustrative retraining loop, not a
replacement for the original, much larger and more carefully validated
training pipeline. Early on, with few accumulated samples, results will be
noisy — that's expected and is reported, not hidden.
"""
import os
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

import db
import predict_model as pm

MODELS_DIR = pm.MODELS_DIR
ARCHIVE_DIR = os.path.join(MODELS_DIR, "archive")
HISTORY_PATH = os.path.join(MODELS_DIR, "retrain_history.json")
CURRENT_MODEL_PATH = os.path.join(MODELS_DIR, "exercise_classifier.pkl")

MIN_SAMPLES_PER_CLASS = 5
MIN_CLASSES = 2


def _load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def _save_history(history):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def get_status():
    """How much labeled data is available, and the retrain history so far."""
    labeled_sets = db.get_labeled_sets_with_readings()
    per_class = {}
    for s in labeled_sets:
        per_class[s["intended_exercise"]] = per_class.get(s["intended_exercise"], 0) + 1

    history = _load_history()
    # Ready as soon as at least MIN_CLASSES exercises individually clear the
    # per-class threshold -- a lone sparse/leftover exercise (e.g. 1-2 stray
    # sets) must not block retraining on classes that are otherwise ready.
    qualifying_classes = [ex for ex, n in per_class.items() if n >= MIN_SAMPLES_PER_CLASS]
    ready = len(qualifying_classes) >= MIN_CLASSES

    return {
        "total_labeled_sets": len(labeled_sets),
        "sets_per_exercise": per_class,
        "ready_to_retrain": ready,
        "min_samples_per_class_required": MIN_SAMPLES_PER_CLASS,
        "min_classes_required": MIN_CLASSES,
        "history": history,
    }

def _build_training_data():
    """Pull every labeled set from exercises that meet the minimum sample
    threshold, re-derive features, return (X, y). Sparse/leftover exercises
    below the threshold are excluded -- including them risks a class with
    too few members to stratify the train/test split, and isn't meaningful
    to train on anyway."""
    labeled_sets = db.get_labeled_sets_with_readings()

    per_class = {}
    for s in labeled_sets:
        per_class[s["intended_exercise"]] = per_class.get(s["intended_exercise"], 0) + 1
    qualifying_classes = {ex for ex, n in per_class.items() if n >= MIN_SAMPLES_PER_CLASS}

    X_parts = []
    y_parts = []
    for s in labeled_sets:
        if s["intended_exercise"] not in qualifying_classes:
            continue
        readings = s["raw_readings"]
        df = pd.DataFrame(readings)
        df.index = pd.to_datetime(df["t"], unit="ms")
        df = df[pm.REQUIRED_RAW_COLUMNS]
        try:
            X_set = pm.extract_features(df)
        except Exception:
            continue  # skip sets too short/malformed to featurize
        X_parts.append(X_set)
        y_parts.extend([s["intended_exercise"]] * len(X_set))

    if not X_parts:
        return None, None

    X = pd.concat(X_parts, ignore_index=True)
    y = np.array(y_parts)
    return X, y


def retrain():
    """
    Retrains on all accumulated labeled data. Returns a result dict with
    before/after accuracy on a held-out split, or raises ValueError if there
    isn't enough data yet.
    """
    status = get_status()
    if not status["ready_to_retrain"]:
        raise ValueError(
            f"Not enough labeled data yet. Need at least {MIN_CLASSES} exercises with "
            f"{MIN_SAMPLES_PER_CLASS}+ sets each. Currently have: {status['sets_per_exercise']}"
        )

    X, y = _build_training_data()
    if X is None or len(X) < 20:
        raise ValueError("Not enough usable feature rows extracted from logged sets to retrain.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Baseline: how the currently-active model does on this same held-out split
    try:
        before_pred = pm._clf.predict(X_test)
        before_acc = round(accuracy_score(y_test, before_pred), 4)
    except Exception:
        before_acc = None  # e.g. class mismatch if exercises differ from original training

    # Train the challenger with the same hyperparameters used for the original model
    new_clf = RandomForestClassifier(
        n_estimators=50, min_samples_leaf=2, criterion="gini", random_state=0
    )
    new_clf.fit(X_train, y_train)
    after_pred = new_clf.predict(X_test)
    after_acc = round(accuracy_score(y_test, after_pred), 4)

    # Version and archive
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    history = _load_history()
    version = len(history) + 1
    timestamp = datetime.now(timezone.utc).isoformat()

    if os.path.exists(CURRENT_MODEL_PATH):
        archive_path = os.path.join(ARCHIVE_DIR, f"exercise_classifier_v{version - 1}.pkl")
        shutil.copy(CURRENT_MODEL_PATH, archive_path)

    joblib.dump(new_clf, CURRENT_MODEL_PATH)

    # Hot-swap: the running process uses the new model immediately, no restart needed
    pm._clf = new_clf

    entry = {
        "version": version,
        "timestamp": timestamp,
        "n_training_samples": len(X_train),
        "n_test_samples": len(X_test),
        "classes": sorted(set(y.tolist())),
        "before_accuracy": before_acc,
        "after_accuracy": after_acc,
        "total_labeled_sets_used": status["total_labeled_sets"],
    }
    history.append(entry)
    _save_history(history)

    return entry