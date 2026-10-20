# ---- Stage 1: build a wheel cache (keeps the final image small & fast) ----
FROM python:3.11-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
# Install only serving deps for a lean runtime image.
RUN pip install --prefix=/install \
      fastapi==0.115.5 "uvicorn[standard]==0.32.1" pydantic==2.10.2 \
      prometheus-client==0.21.1 scikit-learn==1.5.2 pandas==2.2.3 \
      numpy==1.26.4 joblib==1.4.2

# ---- Stage 2: runtime ----
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

# non-root user (security best practice; required by some k8s policies)
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY models/ ./models/          

USER appuser
EXPOSE 8000

# container-level liveness check (k8s probes also hit /health)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
