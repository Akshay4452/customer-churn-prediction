# Slim production image for FastAPI + Streamlit (Phase 4.3)
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

COPY churn_service/ ./churn_service/
COPY app_ui.py .

EXPOSE 8000 8501

CMD ["uvicorn", "churn_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
