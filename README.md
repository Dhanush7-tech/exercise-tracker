# Exercise Tracker

An end-to-end machine learning application that classifies barbell exercises
and counts reps from wearable sensor data — trained, packaged, deployed, and
backed by a persistent database.

![Tests](https://github.com/Dhanush7-tech/exercise-tracker/actions/workflows/tests.yml/badge.svg)

**Live demo:** https://exercise-tracker-sr2d.onrender.com
*(free-tier hosting — the first request after idle time can take 30–50s to wake up)*

![Dashboard screenshot](docs/screenshot-dashboard.png)

## What it does

Given accelerometer + gyroscope readings from a wearable sensor, the model
identifies which of six barbell exercises was performed (bench press,
deadlift, overhead press, row, squat, or rest) and counts the reps in that
set. Since a live sensor isn't connected to the deployed app, "Create a test
dataset" generates new sensor readings — calibrated on real per-exercise
signal statistics (amplitude, dominant axes, rep cadence), not copied from
the training data — so the full pipeline can be demonstrated on genuinely
unseen data.

**Verified results, not just claimed:**
- 99.85% accuracy on a held-out participant the model never trained on
- 98.9% accuracy across 91 real recorded sets run through the full inference pipeline
- 100% accuracy across 80 synthetically generated sets spanning all 6 exercises

## The full story: from raw sensor data to a deployed app

**The problem.** Wearables can record motion, but raw accelerometer/gyroscope
numbers don't tell you what exercise someone did or how many reps they got.
This project builds that missing layer: given a stream of sensor readings,
identify the exercise and count the reps.

**The dataset.** Five participants performed barbell exercises — bench press,
deadlift, overhead press, barbell row, and squat — plus a "rest" class, across
heavy and medium weight categories, wearing a MetaMotion wrist sensor
(accelerometer at 12.5Hz, gyroscope at 25Hz). 187 raw sensor recordings in total.

**Cleaning and feature engineering.** Raw sensor data is noisy and irregularly
sampled, so before any model could learn from it:
- Resampled to a consistent 200ms interval
- Outliers removed via Chauvenet's criterion
- A Butterworth low-pass filter applied to smooth sensor noise while
  preserving the actual movement signal
- PCA to compress the 6 raw axes into their dominant components
- Temporal features (rolling mean/std over each axis) to capture motion
  trends over time
- Frequency features (FFT-based) to capture the rhythmic, periodic nature
  of a repeated lift
- KMeans clustering added as an extra engineered feature

**Model selection.** Multiple algorithms (Neural Network, Random Forest,
KNN, Decision Tree, Naive Bayes) were compared via grid search with forward
feature selection. Random Forest was the strongest performer.

**Results — validated the honest way.** A naive random train/test split
scored 99.9%, but that's inflated: overlapping sensor windows leak between
train and test. The number that actually matters is testing on a
**participant the model never saw during training** — held-out-participant
accuracy came out to **99.85%**.

*(This research pipeline — data ingestion, feature engineering, model
training/evaluation, notebooks — lives in a separate repo:
[Exercise-Tracker-Model-Scratch](https://github.com/Dhanush7-tech/Exercise-Tracker).)*

### From trained model to working application (this repo)

A trained model sitting in a notebook isn't a product. Getting from there to
what's actually deployed here meant:
- **A real inference pipeline** (`predict_model.py`) — the exact same
  preprocessing/feature-engineering steps, frozen and reused on new data,
  since the original pipeline re-fit its transformers on the full dataset
  every run, which doesn't work for a single new prediction. Verified at
  98.9% accuracy across 91 real recorded sets run through it end-to-end.
- **A synthetic data generator** (`synthetic_data.py`) — since a live sensor
  isn't attached to the deployed app, this generates new sensor readings
  calibrated on real per-exercise signal statistics (not copied from
  training data) to demonstrate the pipeline on genuinely unseen input.
  Verified at 100% across 80 generated sets.
- **A FastAPI backend + Postgres/SQLite database** — sessions, logged sets,
  history, analytics.
- **A retraining loop** (`retrain.py`) — logged data can become new labeled
  training examples; retraining evaluates the new model against the current
  one on held-out data before promoting it.
- **Docker + a live deployment** on Render, connected to GitHub for
  auto-deploy.
- **A pytest suite + GitHub Actions CI** — caught and helped fix a real bug
  in the retraining logic before it reached production.

## Architecture

![Architecture diagram](docs/architecture.svg)

The pipeline: raw sensor data → outlier removal → low-pass filtering → PCA +
temporal/frequency feature extraction → clustering → RandomForest
classification → peak-detection rep counting. The same feature-engineering
artifacts (fitted PCA, KMeans, normalization stats) used in training are
frozen and reused at inference time, so new data lands in the exact feature
space the model was trained on.

## Tech stack

| Layer | Tools |
|---|---|
| Model | scikit-learn (RandomForest), scipy (signal processing), pandas/numpy |
| Backend | FastAPI, Python |
| Database | PostgreSQL (Neon) in production, SQLite for local dev — auto-switches via `DATABASE_URL` |
| Frontend | HTML / CSS / vanilla JS, Chart.js |
| Deployment | Docker, Render (auto-deploys from GitHub) |

## Run it locally

```bash
docker compose up --build
```
Open http://127.0.0.1:8000/. Uses SQLite by default — no setup required.

To use Postgres locally instead, create a `.env` file with:
```
DATABASE_URL=postgresql://user:password@host/dbname
```

Full setup, deployment, and API details are in [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).
