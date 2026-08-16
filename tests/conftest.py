"""
Shared pytest fixtures.

Critical setup done here, before any test module imports app code:
1. Force an isolated, temporary SQLite database for the whole test session,
   regardless of any local .env / DATABASE_URL configured for the real app.
   Tests must never write into your real Postgres database.
2. Make sure `import api`, `import db`, etc. resolve to this project's modules
   regardless of what directory pytest is invoked from.
"""
import os
import sys
import shutil
import tempfile

import pytest

# --- 1. Isolate the database BEFORE any app module is imported ---
os.environ.pop("DATABASE_URL", None)
_test_db_fd, _TEST_DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)
os.environ["WORKOUT_DB_PATH"] = _TEST_DB_PATH

# --- 2. Make sure the app package is importable ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
import predict_model as pm  # noqa: E402

db.init_db()


@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_db():
    yield
    try:
        os.remove(_TEST_DB_PATH)
    except OSError:
        pass


@pytest.fixture()
def isolated_retrain_env(tmp_path, monkeypatch):
    """
    Retraining writes model files to disk and swaps the live classifier in
    memory. Tests must never touch the real, shipped model artifacts — this
    fixture points retrain.py at a throwaway copy instead, and restores the
    original classifier afterward so later tests see the real model again.
    """
    import retrain as rt

    real_models_dir = pm.MODELS_DIR
    tmp_models_dir = tmp_path / "models"
    shutil.copytree(real_models_dir, tmp_models_dir)

    original_clf = pm._clf

    monkeypatch.setattr(rt, "MODELS_DIR", str(tmp_models_dir))
    monkeypatch.setattr(rt, "ARCHIVE_DIR", str(tmp_models_dir / "archive"))
    monkeypatch.setattr(rt, "HISTORY_PATH", str(tmp_models_dir / "retrain_history.json"))
    monkeypatch.setattr(rt, "CURRENT_MODEL_PATH", str(tmp_models_dir / "exercise_classifier.pkl"))

    yield rt

    pm._clf = original_clf
