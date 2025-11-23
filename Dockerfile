# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=on \
    APP_HOME=/app

WORKDIR ${APP_HOME}

# System dependencies for OpenCV/FFmpeg and builds
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        git && \
    rm -rf /var/lib/apt/lists/*

# Copy project metadata first for better caching
COPY pyproject.toml README.md README_CN.md ${APP_HOME}/
COPY src ${APP_HOME}/src
COPY scripts ${APP_HOME}/scripts
COPY config ${APP_HOME}/config
COPY docs ${APP_HOME}/docs
COPY docker ${APP_HOME}/docker

RUN pip install --upgrade pip setuptools wheel && \
    pip install .[full]

# Runtime defaults
ENV PIPELINE_CONFIG=/app/config/sample-pipeline.yaml \
    DASHBOARD_CONFIG=/app/config/sample-pipeline.yaml \
    DASHBOARD_HOST=0.0.0.0 \
    DASHBOARD_PORT=8080

COPY . ${APP_HOME}

RUN chmod +x docker/*.sh

CMD ["bash", "-c", "echo 'Provide a command, e.g. docker/run_pipeline.sh' && sleep infinity"]
