#
FROM python:3.11

#
WORKDIR /app

COPY requirements.txt .
COPY wheels /tmp/wheels
RUN pip install --no-cache-dir --upgrade --find-links=/tmp/wheels -r requirements.txt

#
COPY ./app /app

# The image runs offline: it carries no Materials Project or MPDS key, and gliquid raises
# OfflineError for a system the store does not cover.
# GLIQUID_CACHE_DIR names the read-only SQLite store; the .sqlite suffix selects
# cache_mode='sqlite'.
# The pod runs under an arbitrary UID with HOME=/, so MPLCONFIGDIR and XDG_CACHE_HOME put the
# matplotlib and fontconfig caches in /tmp.
# OMP_NUM_THREADS and OPENBLAS_NUM_THREADS give NumPy and SciPy one thread each: the pod's CPU
# limit is far below the node's core count, which the libraries would otherwise size their
# thread pools to.
ENV GLIQUID_OFFLINE=1 \
    GLIQUID_CACHE_DIR=/app/gliqviz/gliquid_cache.sqlite \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

#
EXPOSE 8000

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
