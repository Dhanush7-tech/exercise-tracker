import db


def test_create_session_returns_an_id():
    session_id = db.create_session("pytest session")
    assert isinstance(session_id, int)
    assert session_id > 0


def test_log_set_and_retrieve_from_history():
    session_id = db.create_session("history roundtrip test")
    set_id = db.log_set(
        session_id=session_id, set_number=1, intended_exercise="squat", target_reps=8,
        predicted_exercise="squat", predicted_reps=8, confidence=0.91,
        n_epochs_used=50, filename="test.csv",
    )
    assert isinstance(set_id, int)

    history = db.get_history()
    match = next((h for h in history if h["id"] == set_id), None)
    assert match is not None
    assert match["intended_exercise"] == "squat"
    assert match["predicted_exercise"] == "squat"
    assert match["match"] == 1


def test_match_flag_is_zero_when_prediction_differs_from_intent():
    session_id = db.create_session("mismatch test")
    set_id = db.log_set(
        session_id=session_id, set_number=1, intended_exercise="bench", target_reps=5,
        predicted_exercise="squat", predicted_reps=5, confidence=0.5,
        n_epochs_used=10, filename="x.csv",
    )
    history = db.get_history()
    row = next(h for h in history if h["id"] == set_id)
    assert row["match"] == 0


def test_match_flag_is_none_when_no_intended_exercise_given():
    session_id = db.create_session("no intent test")
    set_id = db.log_set(
        session_id=session_id, set_number=1, intended_exercise=None, target_reps=None,
        predicted_exercise="row", predicted_reps=6, confidence=0.7,
        n_epochs_used=20, filename="x.csv",
    )
    history = db.get_history()
    row = next(h for h in history if h["id"] == set_id)
    assert row["match"] is None


def test_analytics_totals_increase_after_logging():
    before = db.get_analytics()
    session_id = db.create_session("analytics test")
    db.log_set(
        session_id=session_id, set_number=1, intended_exercise="ohp", target_reps=6,
        predicted_exercise="ohp", predicted_reps=6, confidence=0.8,
        n_epochs_used=30, filename="x.csv",
    )
    after = db.get_analytics()
    assert after["total_sets"] == before["total_sets"] + 1
    assert after["total_reps"] == before["total_reps"] + 6


def test_raw_readings_stored_and_retrievable():
    session_id = db.create_session("raw readings test")
    readings = [{"t": 0, "acc_x": 0.1, "acc_y": 0.2, "acc_z": 0.3,
                 "gyro_x": 1.0, "gyro_y": 2.0, "gyro_z": 3.0}]
    db.log_set(
        session_id=session_id, set_number=1, intended_exercise="dead", target_reps=8,
        predicted_exercise="dead", predicted_reps=8, confidence=0.9,
        n_epochs_used=40, filename="x.csv", raw_readings=readings,
    )
    labeled = db.get_labeled_sets_with_readings()
    dead_sets = [s for s in labeled if s["intended_exercise"] == "dead"]
    assert len(dead_sets) >= 1
    assert dead_sets[-1]["raw_readings"] == readings


def test_sets_without_raw_readings_excluded_from_labeled_query():
    session_id = db.create_session("no raw readings test")
    set_id = db.log_set(
        session_id=session_id, set_number=1, intended_exercise="row", target_reps=8,
        predicted_exercise="row", predicted_reps=8, confidence=0.9,
        n_epochs_used=40, filename="x.csv",  # no raw_readings passed
    )
    labeled = db.get_labeled_sets_with_readings()
    assert not any(s["id"] == set_id for s in labeled)
