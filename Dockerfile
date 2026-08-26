# =============================================================================
# Multi-Stage Dockerfile for Supply Chain Machine Learning Platform
# Serves FastAPI REST Backend (:8000) and Streamlit Web Dashboard (:8501)
# =============================================================================

FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase and trained model artifacts
COPY . .

# Expose FastAPI (:8000) and Streamlit (:8501) ports
EXPOSE 8000 8501

# Default command: launch Streamlit portal
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
