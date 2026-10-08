# webui: FastAPI app + visionist client + the committed SPA (web/dist).
# Build from the REPO ROOT (client/ and webui/ are siblings):
#   docker build -f webui/Dockerfile -t sipgisr/visionist-webui:0.1.0 .
# Runtime: fleet data dir is the /data volume (bind ./data); point the fleet
# entries at the box addresses reachable from THIS container (host.docker.internal
# for host-published ports, or compose service names on the same network).
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
COPY webui/web/dist web/dist

ENV PYTHONPATH=/srv/webui/src \
    WEBUI_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8000

ENTRYPOINT ["python", "-m", "uvicorn", "webui.app:factory", "--host", "0.0.0.0", "--port", "8000"]
