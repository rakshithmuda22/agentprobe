FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY .agentprobe/ .agentprobe/

RUN pip install --no-cache-dir . && \
    rm -rf /root/.cache/pip

ENTRYPOINT ["python", "-m", "src.integrations.action_runner"]
