"""
Synthetic sensor-reading generator.

Produces plausible accelerometer/gyroscope CSVs for a chosen exercise, number
of sets, and reps per set -- calibrated against the real per-exercise signal
statistics (mean level, amplitude/std, dominant axes, rep cadence) so the
output is representative of that movement, but it is NOT a copy of any real
recording. This exists to demonstrate the full pipeline (preprocessing,
filtering, feature engineering, prediction) on data the model was not
trained on.
"""
import numpy as np
import pandas as pd

SAMPLE_INTERVAL_MS = 200  # matches the 200ms-resampled training data

# Calibrated from data/interim/01_data_processed.pkl (mean / std per axis)
# and the rep-detection cutoffs already used in count_repetitions.py.
EXERCISE_PROFILES = {
    "bench": {
        "mean": {"acc_x": -0.09, "acc_y": 0.95, "acc_z": -0.16, "gyro_x": 0.73, "gyro_y": -1.94, "gyro_z": 0.76},
        "amp":  {"acc_x": 0.10,  "acc_y": 0.15, "acc_z": 0.13,  "gyro_x": 13.7, "gyro_y": 8.4,   "gyro_z": 15.9},
        "dominant": ["acc_y", "gyro_z"],
        "seconds_per_rep": 2.4,
    },
    "ohp": {
        "mean": {"acc_x": -0.24, "acc_y": 0.92, "acc_z": -0.12, "gyro_x": 0.86, "gyro_y": -2.16, "gyro_z": 0.80},
        "amp":  {"acc_x": 0.11,  "acc_y": 0.19, "acc_z": 0.14,  "gyro_x": 25.5, "gyro_y": 12.8,  "gyro_z": 27.1},
        "dominant": ["acc_y", "gyro_x", "gyro_z"],
        "seconds_per_rep": 2.6,
    },
    "squat": {
        "mean": {"acc_x": 0.15, "acc_y": 0.67, "acc_z": 0.65, "gyro_x": 0.90, "gyro_y": -1.85, "gyro_z": 0.55},
        "amp":  {"acc_x": 0.18, "acc_y": 0.15, "acc_z": 0.19, "gyro_x": 13.0, "gyro_y": 5.0,   "gyro_z": 4.3},
        "dominant": ["acc_z", "acc_y", "gyro_x"],
        "seconds_per_rep": 2.9,
    },
    "dead": {
        "mean": {"acc_x": 0.04, "acc_y": -1.02, "acc_z": -0.01, "gyro_x": 1.08, "gyro_y": -1.90, "gyro_z": 0.58},
        "amp":  {"acc_x": 0.05, "acc_y": 0.17,  "acc_z": 0.17,  "gyro_x": 25.0, "gyro_y": 11.3,  "gyro_z": 6.1},
        "dominant": ["acc_y", "gyro_x"],
        "seconds_per_rep": 2.7,
    },
    "row": {
        "mean": {"acc_x": 0.03, "acc_y": -1.02, "acc_z": 0.03, "gyro_x": 0.52, "gyro_y": -1.95, "gyro_z": 0.57},
        "amp":  {"acc_x": 0.07, "acc_y": 0.27,  "acc_z": 0.12, "gyro_x": 22.7, "gyro_y": 9.7,   "gyro_z": 12.9},
        "dominant": ["acc_y", "gyro_x", "gyro_z"],
        "seconds_per_rep": 1.6,
    },
    "rest": {
        "mean": {"acc_x": 0.50, "acc_y": -0.52, "acc_z": 0.31, "gyro_x": 0.57, "gyro_y": -1.32, "gyro_z": 2.06},
        "amp":  {"acc_x": 0.05, "acc_y": 0.05,  "acc_z": 0.05, "gyro_x": 3.0,  "gyro_y": 3.0,   "gyro_z": 3.0},
        "dominant": [],
        "seconds_per_rep": None,
    },
}

AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def generate_set(exercise: str, reps: int, seed: int = None) -> pd.DataFrame:
    """Generate one synthetic workout set as a DataFrame with a DatetimeIndex
    and columns acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z."""
    if exercise not in EXERCISE_PROFILES:
        raise ValueError(f"Unknown exercise '{exercise}'. Choose from {list(EXERCISE_PROFILES)}")

    rng = np.random.default_rng(seed)
    profile = EXERCISE_PROFILES[exercise]

    if exercise == "rest":
        duration_s = max(reps, 1) * 2  # "reps" doesn't apply to rest; just make a short recording
    else:
        duration_s = reps * profile["seconds_per_rep"]

    n_samples = max(int(duration_s * 1000 / SAMPLE_INTERVAL_MS), 20)
    t = np.arange(n_samples) * (SAMPLE_INTERVAL_MS / 1000.0)

    data = {}
    for axis in AXES:
        mean = profile["mean"][axis]
        amp = profile["amp"][axis]
        series = np.full(n_samples, mean, dtype=float)

        if exercise != "rest" and axis in profile["dominant"]:
            rep_freq = 1.0 / profile["seconds_per_rep"]
            # rep cycle: fundamental + a touch of 2nd harmonic so it isn't a pure sine
            phase = rng.uniform(0, 2 * np.pi)
            cycle = np.sin(2 * np.pi * rep_freq * t + phase)
            cycle += 0.25 * np.sin(2 * np.pi * 2 * rep_freq * t + phase)
            # slight rep-to-rep amplitude variation (fatigue / imperfect form)
            envelope = 1.0 + 0.08 * rng.standard_normal(n_samples)
            # ramp up/down at the start/end of the set
            ramp = np.clip(np.minimum(t, duration_s - t) / max(profile["seconds_per_rep"] * 0.5, 0.1), 0, 1)
            series += amp * cycle * envelope * ramp

        # sensor noise, present on every axis
        noise_scale = amp * 0.25 if (exercise != "rest" and axis in profile["dominant"]) else amp
        series += rng.normal(0, max(noise_scale, 0.02), n_samples)
        data[axis] = series

    df = pd.DataFrame(data)
    df.index = pd.date_range("2000-01-01", periods=n_samples, freq=f"{SAMPLE_INTERVAL_MS}ms")
    return df


def generate_session(exercise: str, num_sets: int, reps_per_set: int, seed: int = None):
    """Generate multiple sets for a session. Returns a list of DataFrames, one per set."""
    rng = np.random.default_rng(seed)
    sets = []
    for i in range(num_sets):
        set_seed = int(rng.integers(0, 1_000_000))
        sets.append(generate_set(exercise, reps_per_set, seed=set_seed))
    return sets


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.reset_index().rename(columns={"index": "timestamp"})
    return out.to_csv(index=False).encode("utf-8")


if __name__ == "__main__":
    import sys
    exercise = sys.argv[1] if len(sys.argv) > 1 else "squat"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    df = generate_set(exercise, reps, seed=42)
    print(df.describe())
    print(f"\nGenerated {len(df)} rows for {reps} reps of {exercise}")
