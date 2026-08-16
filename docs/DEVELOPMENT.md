# Development Guide

## Run with Docker (recommended — this is what actually gets deployed)
```bash
docker compose up --build
```
Then open **http://127.0.0.1:8000/**. Data persists in a named Docker volume
(`workout_data`) across restarts when using SQLite.

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

## Database: SQLite locally, Postgres in production
`db.py` auto-detects which to use via the `DATABASE_URL` environment variable:
- **Not set** → SQLite file.
- **Set** → Postgres, so history/analytics survive redeploys and restarts.

To use Postgres locally, create a `.env` file (already gitignored) in the
project root:
```
DATABASE_URL=postgresql://user:password@host/dbname
```
[Neon](https://neon.tech) and [Supabase](https://supabase.com) both have free
tiers that take a couple of minutes to set up.

No code changes needed to switch — it's purely driven by whether
`DATABASE_URL` is present. `init_db()` creates the tables automatically on
first run against whichever backend is active.

## Deploying (as currently deployed on Render)
1. Push to GitHub (`.env` is gitignored — never commit real credentials).
2. On Render: **New Web Service** → connect the repo → it detects the
   `Dockerfile` automatically.
3. Add an environment variable: `DATABASE_URL` = your Postgres connection string.
4. Deploy. Render sets `$PORT` automatically; the Dockerfile already respects it.

Free tiers on Render/Railway/Fly.io have no persistent disk for plain files —
that's exactly why Postgres (not SQLite) backs the production deployment.

## API endpoints
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serves the dashboard |
| `GET` | `/api/exercises` | List of exercise classes the model knows |
| `POST` | `/api/sessions` | Create a new workout session |
| `POST` | `/api/generate-and-log` | Generate synthetic sensor data for N sets, run through the model, log results |
| `GET` | `/api/history` | All logged sets |
| `GET` | `/api/analytics` | Aggregate stats: totals, per-exercise breakdown, accuracy, reps-by-day |

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
- `db.py` — persistence layer (SQLite locally, Postgres in production)
- `models/` — frozen model + fitted transformers (PCA, KMeans, normalization stats)
- `static/` — the dashboard (index.html, styles.css, app.js)
- `Dockerfile`, `docker-compose.yml` — containerization
