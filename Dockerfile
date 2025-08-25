# Dockerfile (at repo root)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app
# מעתיק את הריפו המקומי שלך (כולל התיקון ל-setup.py) לתוך האימג'
COPY . /app

# התקנות
RUN python -m pip install -U pip setuptools wheel \
 && python -m pip install -e . \
 && python -m pip install faiss-cpu onnxruntime numpy

# נריץ את הבנצ' מתוך התיקייה הנכונה
WORKDIR /app/examples/benchmark
CMD ["python", "benchmark_sqlite_faiss_onnx.py"]
