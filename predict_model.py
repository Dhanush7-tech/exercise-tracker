"""
End-to-end inference: raw sensor readings for ONE workout set -> (exercise label, confidence, rep count)

Expects a DataFrame with a datetime index (or a 'timestamp'/'epoch (ms)' column)
and columns: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
sampled at (or resampled to) 200ms intervals, matching the training pipeline.
"""
import math
import numpy as np
import pandas as pd
import scipy.special
import joblib
import os
from scipy.signal import argrelextrema

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ---- Load all frozen artifacts once ----
_clf = joblib.load(f"{MODELS_DIR}/exercise_classifier.pkl")
_feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
_lowpass_params = joblib.load(f"{MODELS_DIR}/lowpass_params.pkl")
_norm_stats = joblib.load(f"{MODELS_DIR}/pca_norm_stats.pkl")
_pca = joblib.load(f"{MODELS_DIR}/pca_model.pkl")
_kmeans = joblib.load(f"{MODELS_DIR}/kmeans_model.pkl")
_predictor_columns = joblib.load(f"{MODELS_DIR}/predictor_columns.pkl")  # acc_x..gyro_z

REQUIRED_RAW_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# Same low-pass filter implementation used at training time
from scipy.signal import butter, filtfilt

def _low_pass_filter(data_table, col, sampling_frequency, cutoff_frequency, order=5):
    nyq = 0.5 * sampling_frequency
    cut = cutoff_frequency / nyq
    b, a = butter(order, cut, btype="low", output="ba", analog=False)
    data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
    return data_table


def _mark_outliers_chauvenet(dataset, col, C=2):
    dataset = dataset.copy()
    mean = dataset[col].mean()
    std = dataset[col].std()
    if std == 0 or np.isnan(std):
        dataset[col + "_outlier"] = False
        return dataset
    N = len(dataset.index)
    criterion = 1.0 / (C * N)
    deviation = abs(dataset[col] - mean) / std
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    mask = []
    for i in range(len(dataset.index)):
        prob = 1.0 - 0.5 * (scipy.special.erf(high.iloc[i]) - scipy.special.erf(low.iloc[i]))
        mask.append(prob < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


def _remove_outliers(df):
    """
    NOTE: training removed outliers per-label using label-grouped mean/std.
    At inference time the label is unknown (that's what we're predicting),
    so we approximate by computing Chauvenet stats within this single set.
    This is a reasonable approximation since a set is already one homogeneous
    movement, but it is a documented simplification vs. the training pipeline.
    """
    df = df.copy()
    for col in REQUIRED_RAW_COLUMNS:
        marked = _mark_outliers_chauvenet(df, col)
        df.loc[marked[col + "_outlier"], col] = np.nan
    df[REQUIRED_RAW_COLUMNS] = df[REQUIRED_RAW_COLUMNS].interpolate()
    df[REQUIRED_RAW_COLUMNS] = df[REQUIRED_RAW_COLUMNS].bfill().ffill()
    return df


def _resample_to_200ms(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex (or pass one via load_raw_csv).")
    sampling = {c: "mean" for c in REQUIRED_RAW_COLUMNS}
    return df.resample("200ms").apply(sampling).dropna()


def load_raw_csv(path, timestamp_col=None):
    """Load a CSV export into the shape the pipeline expects."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if timestamp_col is None:
        for cand in ["epoch (ms)", "timestamp", "time"]:
            if cand in df.columns:
                timestamp_col = cand
                break

    if timestamp_col == "epoch (ms)":
        df.index = pd.to_datetime(df[timestamp_col], unit="ms")
    elif timestamp_col is not None:
        df.index = pd.to_datetime(df[timestamp_col])
    else:
        # No timestamp column: assume already evenly spaced, synthesize one at 200ms
        df.index = pd.date_range("2000-01-01", periods=len(df), freq="200ms")

    return df[REQUIRED_RAW_COLUMNS]


def _build_features(df_clean):
    df_lowpass = df_clean.copy()
    fs = _lowpass_params["fs"]
    cutoff = _lowpass_params["cutoff"]
    order = _lowpass_params["order"]

    for col in _predictor_columns:
        df_lowpass = _low_pass_filter(df_lowpass, col, fs, cutoff, order=order)
        df_lowpass[col] = df_lowpass[col + "_lowpass"]
        del df_lowpass[col + "_lowpass"]

    # PCA using saved normalization stats + fitted PCA (transform only, no refit)
    dt_norm = df_lowpass.copy()
    for col in _predictor_columns:
        stats = _norm_stats[col]
        denom = (stats["max"] - stats["min"]) or 1e-9
        dt_norm[col] = (df_lowpass[col] - stats["mean"]) / denom

    pca_vals = _pca.transform(dt_norm[_predictor_columns])
    df_feat = df_lowpass.copy()
    for i in range(pca_vals.shape[1]):
        df_feat[f"pca_{i+1}"] = pca_vals[:, i]

    df_feat["acc_r"] = np.sqrt(df_feat["acc_x"]**2 + df_feat["acc_y"]**2 + df_feat["acc_z"]**2)
    df_feat["gyro_r"] = np.sqrt(df_feat["gyro_x"]**2 + df_feat["gyro_y"]**2 + df_feat["gyro_z"]**2)

    # Duration feature (seconds), same definition as training
    duration = (df_feat.index[-1] - df_feat.index[0]).total_seconds()
    df_feat["duration"] = duration

    # Temporal abstraction: rolling mean/std, window size 5
    temporal_cols = list(_predictor_columns) + ["acc_r", "gyro_r"]
    ws = 5
    for col in temporal_cols:
        df_feat[f"{col}_temp_mean_ws_{ws}"] = df_feat[col].rolling(ws).apply(np.mean)
        df_feat[f"{col}_temp_std_ws_{ws}"] = df_feat[col].rolling(ws).apply(np.std)

    # Frequency abstraction: FFT window size 14, sampling rate 5Hz
    df_freq = df_feat.reset_index(drop=True)
    fs_freq = int(1000 / 200)
    ws_freq = int(2800 / 200)
    freqs = np.round((np.fft.rfftfreq(ws_freq) * fs_freq), 3)

    for col in temporal_cols:
        df_freq[f"{col}_max_freq"] = np.nan
        df_freq[f"{col}_freq_weighted"] = np.nan
        df_freq[f"{col}_pse"] = np.nan
        for freq in freqs:
            df_freq[f"{col}_freq_{freq}_Hz_ws_{ws_freq}"] = np.nan

    n = len(df_freq)
    for i in range(ws_freq, n):
        for col in temporal_cols:
            window = df_freq[col].iloc[i - ws_freq : min(i + 1, n)]
            transformation = np.fft.rfft(window, len(window))
            real_ampl = transformation.real
            for j, freq in enumerate(freqs):
                df_freq.loc[i, f"{col}_freq_{freq}_Hz_ws_{ws_freq}"] = real_ampl[j]
            df_freq.loc[i, f"{col}_max_freq"] = freqs[np.argmax(real_ampl)]
            df_freq.loc[i, f"{col}_freq_weighted"] = float(np.sum(freqs * real_ampl)) / np.sum(real_ampl)
            PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
            PSD_pdf = np.divide(PSD, np.sum(PSD))
            df_freq.loc[i, f"{col}_pse"] = -np.sum(np.log(PSD_pdf) * PSD_pdf)

    df_freq = df_freq.dropna().reset_index(drop=True)

    if len(df_freq) == 0:
        raise ValueError(
            f"Set too short to extract frequency features (need > {ws_freq} rows "
            f"= {ws_freq * 0.2:.1f}s of data at 200ms sampling). Got {n} rows."
        )

    # Clustering using saved KMeans (predict, not fit)
    df_freq["cluster"] = _kmeans.predict(df_freq[["acc_x", "acc_y", "acc_z"]])

    return df_lowpass, df_freq


def _count_reps(df_lowpass_full, exercise_label):
    """Rep counting via peak detection, mirroring count_repetitions.py (with the
    row-exercise column bug fixed: it must use gyro_x, not acc_r, for rows)."""
    fs = _lowpass_params["fs"]
    column, cutoff, order = "acc_r", 0.4, 10

    df = df_lowpass_full.copy()
    df["acc_r"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["gyro_r"] = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2)

    if exercise_label == "squat":
        cutoff = 0.35
    elif exercise_label == "row":
        cutoff, column = 0.65, "gyro_x"
    elif exercise_label == "ohp":
        cutoff = 0.35
    elif exercise_label == "rest":
        return 0

    data = _low_pass_filter(df, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order)
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    return int(len(indexes[0]))


def predict(df_raw):
    """
    df_raw: DataFrame with DatetimeIndex and columns acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
            (use load_raw_csv() to build this from a CSV export).
    Returns a dict: exercise, confidence, reps, per_epoch_predictions, n_epochs_used
    """
    df_resampled = _resample_to_200ms(df_raw)
    df_clean = _remove_outliers(df_resampled)
    df_lowpass, df_feat = _build_features(df_clean)

    X = df_feat[_feature_columns]
    row_preds = _clf.predict(X)
    row_proba = _clf.predict_proba(X)

    # Majority vote across epochs in the set = the set's exercise label
    labels, counts = np.unique(row_preds, return_counts=True)
    exercise = labels[np.argmax(counts)]
    confidence = float(np.mean(row_proba[:, list(_clf.classes_).index(exercise)]))

    reps = _count_reps(df_lowpass, exercise)

    return {
        "exercise": exercise,
        "confidence": round(confidence, 4),
        "reps": reps,
        "n_epochs_used": int(len(df_feat)),
        "epoch_vote_breakdown": dict(zip(labels.tolist(), counts.tolist())),
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        df = load_raw_csv(path)
        print(predict(df))
