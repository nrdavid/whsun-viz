#
FROM python:3.11

#
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

#
COPY ./app /app

# OFFLINE BY CONSTRUCTION, not by omission. This image carries no Materials Project
# and no MPDS key; gliquid's remote paths call config.require_online() BEFORE any
# credential lookup or client construction, so an uncovered system raises a named
# OfflineError instead of issuing a request that would fail obscurely.
# gliquid_app.py also calls set_offline(True) at import, so clearing this variable at
# `docker run` does not silently re-enable the network.
# GLIQUID_CACHE_DIR names the single read-only SQLite store that replaced the
# 4,991-file binary_cache/ tree; a .sqlite suffix also selects cache_mode='sqlite'.
# gliqviz/ holds both the code and the data served at /gliquid/. It may not be named
# `gliquid` -- /app is the WORKDIR and first on sys.path, so that would shadow the
# installed package.
ENV GLIQUID_OFFLINE=1 \
    GLIQUID_CACHE_DIR=/app/gliqviz/gliquid_cache.sqlite

#
EXPOSE 8000

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
