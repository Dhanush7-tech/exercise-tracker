# Exercise Tracker API — container image
FROM python:3.11-slim

WORKDIR /app

# No apt-get / build-essential needed: scipy and scikit-learn ship precompiled
# wheels for this Python version, so pip installs them without compiling anything.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persist the SQLite DB outside the image layer (mount a volume here in production)
RUN mkdir -p /app/data
ENV WORKOUT_DB_PATH=/app/data/workout.db

# Most hosts (Render, Railway, Fly.io) inject $PORT; default to 8000 for local `docker run`
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
