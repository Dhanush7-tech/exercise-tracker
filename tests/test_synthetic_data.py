import synthetic_data as sd


def test_generates_all_known_exercises():
    for exercise in sd.EXERCISE_PROFILES:
        df = sd.generate_set(exercise, reps=6, seed=1)
        assert len(df) > 10
        assert list(df.columns) == sd.AXES
        assert not df.isnull().values.any()


def test_more_reps_produces_a_longer_recording():
    short = sd.generate_set("squat", reps=4, seed=1)
    long = sd.generate_set("squat", reps=12, seed=1)
    assert len(long) > len(short)


def test_same_seed_is_reproducible():
    a = sd.generate_set("bench", reps=8, seed=42)
    b = sd.generate_set("bench", reps=8, seed=42)
    assert (a.values == b.values).all()


def test_different_seeds_produce_different_data():
    a = sd.generate_set("bench", reps=8, seed=1)
    b = sd.generate_set("bench", reps=8, seed=2)
    assert not (a.values == b.values).all()


def test_unknown_exercise_raises():
    import pytest
    with pytest.raises(ValueError):
        sd.generate_set("not_a_real_exercise", reps=8, seed=1)


def test_generate_session_returns_requested_number_of_sets():
    sets = sd.generate_session("row", num_sets=4, reps_per_set=8, seed=1)
    assert len(sets) == 4
    for s in sets:
        assert list(s.columns) == sd.AXES
