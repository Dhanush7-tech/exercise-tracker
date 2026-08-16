"""
Persistence layer for the Exercise Tracker app.

- Locally (no DATABASE_URL set): uses SQLite, a file on disk.
- In production (DATABASE_URL set, e.g. by Render/Railway/Neon/Supabase):
  uses Postgres, so history/analytics survive redeploys and restarts.

Every uploaded/generated set gets logged here, which becomes the app's
growing workout history / dataset. Sets logged with a known intended_exercise
also store their raw sensor readings (as JSON), which the retraining pipeline
uses as new labeled training examples.
"""
import os
import json
from datetime import datetime, timezone

DATABASE_URL = os.environ.get("DATABASE_URL")
IS_POSTGRES = bool(DATABASE_URL)

if IS_POSTGRES:
    import psycopg2
    import psycopg2.extras
else:
    import sqlite3

# Only used for local SQLite; ignored when DATABASE_URL is set.
DB_PATH = os.environ.get(
    "WORKOUT_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "workout.db"),
)


def get_conn():
    if IS_POSTGRES:
        return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def _q(sql: str) -> str:
    """Translate sqlite-style '?' placeholders to psycopg2-style '%s' when on Postgres."""
    return sql.replace("?", "%s") if IS_POSTGRES else sql


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    if IS_POSTGRES:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                label TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sets (
                id SERIAL PRIMARY KEY,
                session_id INTEGER REFERENCES sessions(id),
                set_number INTEGER,
                intended_exercise TEXT,
                target_reps INTEGER,
                predicted_exercise TEXT NOT NULL,
                predicted_reps INTEGER NOT NULL,
                confidence REAL NOT NULL,
                n_epochs_used INTEGER,
                match INTEGER,
                filename TEXT,
                created_at TEXT NOT NULL,
                raw_readings TEXT
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                set_number INTEGER,
                intended_exercise TEXT,
                target_reps INTEGER,
                predicted_exercise TEXT NOT NULL,
                predicted_reps INTEGER NOT NULL,
                confidence REAL NOT NULL,
                n_epochs_used INTEGER,
                match INTEGER,
                filename TEXT,
                created_at TEXT NOT NULL,
                raw_readings TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)

    conn.commit()
    conn.close()

    # Best-effort migration for databases created before raw_readings existed
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("ALTER TABLE sets ADD COLUMN raw_readings TEXT")
        conn.commit()
        conn.close()
    except Exception:
        pass


def create_session(label):
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    if IS_POSTGRES:
        cur.execute(_q("INSERT INTO sessions (label, created_at) VALUES (?, ?) RETURNING id"), (label, now))
        session_id = cur.fetchone()["id"]
    else:
        cur.execute(_q("INSERT INTO sessions (label, created_at) VALUES (?, ?)"), (label, now))
        session_id = cur.lastrowid

    conn.commit()
    conn.close()
    return session_id


def log_set(session_id, set_number, intended_exercise, target_reps,
            predicted_exercise, predicted_reps, confidence, n_epochs_used, filename,
            raw_readings=None):
    match = None
    if intended_exercise:
        match = 1 if intended_exercise == predicted_exercise else 0

    raw_json = json.dumps(raw_readings) if raw_readings is not None else None

    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    sql = """
        INSERT INTO sets (session_id, set_number, intended_exercise, target_reps,
                           predicted_exercise, predicted_reps, confidence, n_epochs_used,
                           match, filename, created_at, raw_readings)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (session_id, set_number, intended_exercise, target_reps,
              predicted_exercise, predicted_reps, confidence, n_epochs_used,
              match, filename, now, raw_json)

    if IS_POSTGRES:
        cur.execute(_q(sql + " RETURNING id"), params)
        set_id = cur.fetchone()["id"]
    else:
        cur.execute(_q(sql), params)
        set_id = cur.lastrowid

    conn.commit()
    conn.close()
    return set_id


def get_history(limit=200):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(_q("""
        SELECT sets.*, sessions.label as session_label
        FROM sets
        LEFT JOIN sessions ON sets.session_id = sessions.id
        ORDER BY sets.created_at DESC
        LIMIT ?
    """), (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_labeled_sets_with_readings():
    """Sets that have both a known intended_exercise (ground-truth label) and
    stored raw readings — i.e. usable as new labeled training examples."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, intended_exercise, raw_readings, created_at
        FROM sets
        WHERE intended_exercise IS NOT NULL AND raw_readings IS NOT NULL
        ORDER BY created_at ASC
    """)
    rows = cur.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["raw_readings"] = json.loads(d["raw_readings"])
        result.append(d)
    return result


def get_analytics():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) as c FROM sets")
    total_sets = cur.fetchone()["c"]

    cur.execute("SELECT COALESCE(SUM(predicted_reps),0) as s FROM sets")
    total_reps = cur.fetchone()["s"]

    cur.execute("SELECT COUNT(*) as c FROM sessions")
    total_sessions = cur.fetchone()["c"]

    cur.execute("""
        SELECT predicted_exercise as exercise, COUNT(*) as sets, SUM(predicted_reps) as reps,
               AVG(confidence) as avg_confidence
        FROM sets GROUP BY predicted_exercise ORDER BY sets DESC
    """)
    by_exercise = cur.fetchall()

    cur.execute("""
        SELECT COUNT(*) as total, SUM(match) as matched
        FROM sets WHERE match IS NOT NULL
    """)
    accuracy_row = cur.fetchone()

    cur.execute("""
        SELECT substr(created_at, 1, 10) as day, COUNT(*) as sets, SUM(predicted_reps) as reps
        FROM sets GROUP BY day ORDER BY day ASC
    """)
    by_day = cur.fetchall()

    conn.close()

    accuracy = None
    if accuracy_row["total"]:
        accuracy = round((accuracy_row["matched"] or 0) / accuracy_row["total"], 4)

    return {
        "total_sets": total_sets,
        "total_reps": total_reps or 0,
        "total_sessions": total_sessions,
        "by_exercise": [dict(r) for r in by_exercise],
        "intended_vs_predicted_accuracy": accuracy,
        "matched_count": accuracy_row["matched"] or 0,
        "compared_count": accuracy_row["total"] or 0,
        "by_day": [dict(r) for r in by_day],
    }
