# Exercise Tracker

An end-to-end machine learning application that classifies barbell exercises
and counts reps from wearable sensor data — trained, packaged, deployed, and
backed by a persistent database.

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

## Project background

Built on top of the ["Tracking Barbell Exercises"](https://github.com/Dhanush7-tech/Exercise-Tracker)
data science pipeline (data ingestion, feature engineering, model training/evaluation).
This repo takes that trained model the rest of the way: a real inference
pipeline, an API, a database-backed dashboard, containerization, and a live
deployment — the parts that turn a trained model into a working application.
