FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# Non-root user
RUN useradd -m -u 1000 appuser
ENV PATH="/home/appuser/.local/bin:$PATH"
WORKDIR /app

# Minimal OS deps (needed by scikit-learn / numpy)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy code
COPY --chown=appuser:appuser . .
USER appuser

# Python deps (CPU-only)
# Make sure your requirements.txt includes: flask, gunicorn, pandas, numpy, scikit-learn
RUN pip install --no-cache-dir --user -r requirements.txt

# (Optional) quick smoke test: prints versions and verifies imports
RUN python - <<'PY'
import sys, numpy, pandas, sklearn
print("PY:", sys.version)
print("numpy:", numpy.__version__, "pandas:", pandas.__version__, "sklearn:", sklearn.__version__)
PY

# Expose & run (HF Spaces injects $PORT)
ENV PORT=7860
EXPOSE 7860
CMD ["bash","-lc","gunicorn -w 1 -k gthread --threads 4 --timeout 120 -b 0.0.0.0:${PORT:-7860} app:app"]
