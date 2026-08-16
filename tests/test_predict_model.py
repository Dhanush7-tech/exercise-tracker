import pytest

import predict_model as pm
import synthetic_data as sd


@pytest.mark.parametrize("exercise", list(sd.EXERCISE_PROFILES.keys()))
def test_predicts_correct_exercise_on_calibrated_synthetic_data(exercise):
    """
    The synthetic generator is calibrated on real per-exercise signal
    statistics, so a correctly-working pipeline should classify its own
    output correctly. This is the same check done manually throughout
    development (verified at 100% across 80 generated sets); this test
    formalizes a slice of that.
    """
    reps = 4 if exercise == "rest" else 8
    df = sd.generate_set(exercise, reps=reps, seed=123)
    result = pm.predict(df)

    assert result["exercise"] == exercise
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["reps"] >= 0
    assert result["n_epochs_used"] > 0


def test_extract_features_matches_classifier_expected_columns():
    df = sd.generate_set("squat", reps=8, seed=1)
    X = pm.extract_features(df)
    assert list(X.columns) == pm._feature_columns
    assert len(X) > 0


def test_predict_and_extract_features_are_consistent():
    """predict() internally calls the same feature-building path as
    extract_features(); their row counts should match for the same input."""
    df = sd.generate_set("bench", reps=8, seed=5)
    X = pm.extract_features(df)
    result = pm.predict(df)
    assert result["n_epochs_used"] == len(X)


def test_rejects_recording_too_short_to_featurize():
    df = sd.generate_set("squat", reps=8, seed=1).iloc[:5]
    with pytest.raises(ValueError):
        pm.predict(df)


def test_load_raw_csv_missing_columns_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("acc_x,acc_y\n0.1,0.2\n")
    with pytest.raises(ValueError):
        pm.load_raw_csv(str(bad_csv))
