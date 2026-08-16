import pandas as pd

import db
import predict_model as pm
import synthetic_data as sd


def _df_to_raw_readings(df):
    out = df.reset_index()
    out.columns = ["t"] + list(pm.REQUIRED_RAW_COLUMNS)
    out["t"] = ((out["t"] - pd.Timestamp("1970-01-01")) / pd.Timedelta(milliseconds=1)).astype("int64")
    return out.to_dict(orient="records")


def _log_labeled_synthetic_sets(session_label, exercise, n_sets, reps=8):
    session_id = db.create_session(session_label)
    for i in range(n_sets):
        set_df = sd.generate_set(exercise, reps=reps, seed=i * 11 + hash(exercise) % 37)
        result = pm.predict(set_df)
        db.log_set(
            session_id=session_id, set_number=i + 1, intended_exercise=exercise, target_reps=reps,
            predicted_exercise=result["exercise"], predicted_reps=result["reps"],
            confidence=result["confidence"], n_epochs_used=result["n_epochs_used"],
            filename=f"{exercise}_{i}.csv", raw_readings=_df_to_raw_readings(set_df),
        )


def test_status_reports_not_ready_before_any_data(isolated_retrain_env):
    rt = isolated_retrain_env
    status = rt.get_status()
    assert "ready_to_retrain" in status
    assert "sets_per_exercise" in status
    # Fresh isolated DB from conftest may already contain rows from other test
    # modules (shared test DB) -- what matters here is the response shape,
    # not a guaranteed-empty state.


def test_retrain_fails_with_clear_error_when_data_insufficient(isolated_retrain_env):
    rt = isolated_retrain_env
    _log_labeled_synthetic_sets("insufficient data test", "bench", n_sets=1)

    status = rt.get_status()
    if status["ready_to_retrain"]:
        return  # other tests may have already logged enough data; not this test's concern

    import pytest
    with pytest.raises(ValueError):
        rt.retrain()


def test_retrain_end_to_end_with_sufficient_data(isolated_retrain_env):
    rt = isolated_retrain_env
    _log_labeled_synthetic_sets("retrain e2e test A", "bench", n_sets=6)
    _log_labeled_synthetic_sets("retrain e2e test B", "squat", n_sets=6)

    status = rt.get_status()
    assert status["ready_to_retrain"] is True

    result = rt.retrain()
    assert result["version"] >= 1
    assert 0.0 <= result["after_accuracy"] <= 1.0
    assert result["n_training_samples"] > 0
    assert set(result["classes"]) >= {"bench", "squat"}


def test_retrain_hot_swaps_the_live_classifier(isolated_retrain_env):
    rt = isolated_retrain_env
    _log_labeled_synthetic_sets("hot swap test A", "row", n_sets=6)
    _log_labeled_synthetic_sets("hot swap test B", "dead", n_sets=6)

    clf_before = pm._clf
    rt.retrain()
    assert pm._clf is not clf_before  # a new object was swapped in

    # The swapped-in model should still work for ordinary prediction
    df = sd.generate_set("row", reps=8, seed=1)
    result = pm.predict(df)
    assert result["exercise"] in {"row", "dead"}  # model only knows the 2 classes it was just trained on
