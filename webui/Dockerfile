# webui: FastAPI app + visionist client + the SPA (built in stage 1 from webui/web).
# Build from the REPO ROOT (client/ and webui/ are siblings):
#   docker build -f webui/Dockerfile -t sipgisr/visionist-webui:latest .
# Runtime: fleet data dir is the /data volume (bind ./data); point the fleet
# entries at the box addresses reachable from THIS container (host.docker.internal
# for host-published ports, or compose service names on the same network).
# ---- stage 1: build the SPA (web/dist is gitignored, so it must be built here) ----
FROM node:20-slim AS web
WORKDIR /web
COPY webui/web/package.json webui/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY webui/web/ ./
RUN npm run build            # tsc --noEmit && vite build -> /web/dist

# ---- stage 2: the app ----
FROM python:3.13-slim

WORKDIR /srv
# Pure-python gRPC client (generated pb2 included).  Not on PyPI, so install
# from source.
COPY visionist_client/ /srv/visionist_client/
RUN pip install --no-cache-dir /srv/visionist_client

WORKDIR /srv/webui
COPY webui/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY webui/src src
COPY webui/boxes boxes
COPY webui/pipelines pipelines
COPY --from=web /web/dist web/dist

ENV PYTHONPATH=/srv/webui/src \
    WEBUI_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8000

ENTRYPOINT ["python", "-m", "uvicorn", "webui.app:factory", "--host", "0.0.0.0", "--port", "8000"]
