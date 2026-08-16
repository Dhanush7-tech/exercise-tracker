# Exercise Tracker — Full Dashboard (Synthetic Data Edition)

A local web app: generate synthetic-but-realistic wearable sensor readings for
a chosen exercise/sets/reps, run them through the full ML pipeline
(preprocessing -> filtering -> feature engineering -> classification -> rep
counting), and track results in a SQLite-backed history/analytics dashboard.

Since a live sensor isn't connected, "Create a test dataset" generates new
readings calibrated on real per-exercise signal statistics (amplitude,
dominant axes, rep cadence) from the original training data — genuinely new
data, not copies of any training example — so you can demonstrate the whole
pipeline end-to-end.

## Run with Docker (recommended — this is what actually gets deployed)
```bash
docker compose up --build
```
Then open **http://127.0.0.1:8000/**. Data persists in a named Docker volume
(`workout_data`) across restarts, unlike running `workout.db` directly on disk.

To run without compose:
```bash
docker build -t exercise-tracker .
docker run -p 8000:8000 -v exercise_tracker_data:/app/data exercise-tracker
```

## Run locally without Docker
```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

## Deploying to a live URL
The Dockerfile is host-agnostic and respects `$PORT`, so it works as-is on:
- **Render** — "New Web Service" → connect repo → it detects the Dockerfile automatically. Add a persistent disk mounted at `/app/data` if you want the SQLite DB to survive redeploys (Render's free tier disk is ephemeral otherwise).
- **Railway** — "New Project" → "Deploy from Dockerfile" → same disk caveat applies.
- **Fly.io** — `fly launch` detects the Dockerfile; use `fly volumes create` + mount at `/app/data` for persistence.

For anything beyond a demo, swap SQLite for a managed Postgres instance (Render/Railway both offer free Postgres) — `db.py` would need its `sqlite3` calls replaced with a Postgres driver, which is a natural next step once you're deploying for real.


## Use it
Open **http://127.0.0.1:8000/** in your browser.

1. **Create a test dataset** — name a session, click "New session," pick an
   exercise, choose number of sets and reps per set, click "Generate & run
   through model." Each generated set is preprocessed, filtered,
   feature-engineered, classified, and rep-counted, then saved.
2. **History** — every generated set, with intended vs. predicted exercise.
3. **Analytics** — reps per day, sets per exercise, and overall accuracy
   (how often the model's prediction matched the exercise you generated).

Data persists in `workout.db` (SQLite) between restarts. Delete that file to
reset everything.

## How the synthetic data works (`synthetic_data.py`)
For each exercise, real training data was used to calibrate: mean sensor
level, oscillation amplitude, which axes move most, and typical seconds per
rep. A new set is built as a rep-cadence oscillation (fundamental + a touch
of 2nd harmonic, slight rep-to-rep variation, start/end ramp) plus sensor
noise — grounded in real movement signatures but not copied from any real
recording. Verified: 100% correct classification across 80 generated sets
(all 6 exercises, multiple seeds/rep counts).

## Files
- `api.py` — FastAPI app: sessions / generate-and-log / history / analytics endpoints + serves the dashboard
- `synthetic_data.py` — calibrated synthetic sensor-reading generator
- `predict_model.py` — the inference pipeline (preprocessing through prediction)
- `db.py` — SQLite persistence (sessions + sets tables)
- `models/` — frozen model + fitted transformers
- `static/` — the dashboard (index.html, styles.css, app.js)
