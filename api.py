"""
Exercise Tracker API

Run locally:
    pip install -r requirements.txt
    uvicorn api:app --reload --port 8000

Open the dashboard:
    http://127.0.0.1:8000/

Interactive API docs:
    http://127.0.0.1:8000/docs
"""
import io
import os
import warnings

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

import predict_model as pm
import db
import synthetic_data as sd

warnings.filterwarnings("ignore")

db.init_db()

app = FastAPI(
    title="Exercise Tracker API",
    description="Log workout sets from wearable sensor data, get automatic exercise "
                 "classification + rep counts, and track history/analytics over time.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SetLogResponse(BaseModel):
    id: int
    session_id: int
    set_number: int
    intended_exercise: Optional[str]
    target_reps: Optional[int]
    predicted_exercise: str
    predicted_reps: int
    confidence: float
    n_epochs_used: int
    match: Optional[bool]


class SessionCreateRequest(BaseModel):
    label: Optional[str] = None


def _df_from_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    missing = [c for c in pm.REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {missing}. Required: {pm.REQUIRED_RAW_COLUMNS}",
        )

    timestamp_col = None
    for cand in ["epoch (ms)", "timestamp", "time"]:
        if cand in df.columns:
            timestamp_col = cand
            break

    if timestamp_col == "epoch (ms)":
        df.index = pd.to_datetime(df[timestamp_col], unit="ms")
    elif timestamp_col is not None:
        df.index = pd.to_datetime(df[timestamp_col])
    else:
        df.index = pd.date_range("2000-01-01", periods=len(df), freq="200ms")

    return df[pm.REQUIRED_RAW_COLUMNS]


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Exercise Tracker API is running."}


@app.get("/api/exercises")
def list_exercises():
    return {"exercises": list(pm._clf.classes_)}


@app.post("/api/sessions")
def create_session(payload: SessionCreateRequest):
    label = payload.label or "Workout session"
    session_id = db.create_session(label)
    return {"session_id": session_id, "label": label}


@app.post("/api/log-set", response_model=SetLogResponse)
async def log_set(
    file: UploadFile = File(...),
    session_id: int = Form(...),
    set_number: int = Form(1),
    intended_exercise: Optional[str] = Form(None),
    target_reps: Optional[int] = Form(None),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw_bytes = await file.read()
    df = _df_from_csv_bytes(raw_bytes)

    try:
        result = pm.predict(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    set_id = db.log_set(
        session_id=session_id,
        set_number=set_number,
        intended_exercise=intended_exercise,
        target_reps=target_reps,
        predicted_exercise=result["exercise"],
        predicted_reps=result["reps"],
        confidence=result["confidence"],
        n_epochs_used=result["n_epochs_used"],
        filename=file.filename,
    )

    match = None
    if intended_exercise:
        match = intended_exercise == result["exercise"]

    return {
        "id": set_id,
        "session_id": session_id,
        "set_number": set_number,
        "intended_exercise": intended_exercise,
        "target_reps": target_reps,
        "predicted_exercise": result["exercise"],
        "predicted_reps": result["reps"],
        "confidence": result["confidence"],
        "n_epochs_used": result["n_epochs_used"],
        "match": match,
    }


class GenerateRequest(BaseModel):
    session_id: int
    exercise: str
    num_sets: int
    reps_per_set: int
    starting_set_number: int = 1


@app.post("/api/generate-and-log")
def generate_and_log(payload: GenerateRequest):
    if payload.exercise not in sd.EXERCISE_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown exercise '{payload.exercise}'. Choose from {list(sd.EXERCISE_PROFILES)}",
        )
    if payload.num_sets < 1 or payload.num_sets > 20:
        raise HTTPException(status_code=400, detail="num_sets must be between 1 and 20.")
    if payload.reps_per_set < 1 or payload.reps_per_set > 30:
        raise HTTPException(status_code=400, detail="reps_per_set must be between 1 and 30.")

    synthetic_sets = sd.generate_session(payload.exercise, payload.num_sets, payload.reps_per_set)

    results = []
    for i, df in enumerate(synthetic_sets):
        set_number = payload.starting_set_number + i
        try:
            result = pm.predict(df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed on generated set {i+1}: {e}")

        set_id = db.log_set(
            session_id=payload.session_id,
            set_number=set_number,
            intended_exercise=payload.exercise,
            target_reps=payload.reps_per_set,
            predicted_exercise=result["exercise"],
            predicted_reps=result["reps"],
            confidence=result["confidence"],
            n_epochs_used=result["n_epochs_used"],
            filename=f"synthetic_{payload.exercise}_set{set_number}.csv",
        )
        results.append({
            "id": set_id,
            "set_number": set_number,
            "intended_exercise": payload.exercise,
            "target_reps": payload.reps_per_set,
            "predicted_exercise": result["exercise"],
            "predicted_reps": result["reps"],
            "confidence": result["confidence"],
            "n_epochs_used": result["n_epochs_used"],
            "match": payload.exercise == result["exercise"],
            "csv_preview_rows": len(df),
        })

    return {"session_id": payload.session_id, "generated": results}


@app.get("/api/generate-preview-csv")
def generate_preview_csv(exercise: str, reps: int, seed: Optional[int] = None):
    """Generate one synthetic set and return it as a downloadable CSV, without logging it."""
    if exercise not in sd.EXERCISE_PROFILES:
        raise HTTPException(status_code=400, detail=f"Unknown exercise '{exercise}'.")
    df = sd.generate_set(exercise, reps, seed=seed)
    csv_bytes = sd.to_csv_bytes(df)
    from fastapi.responses import Response
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="synthetic_{exercise}.csv"'},
    )


@app.get("/api/history")
def history(limit: int = 200):
    return {"sets": db.get_history(limit=limit)}


@app.get("/api/analytics")
def analytics():
    return db.get_analytics()
