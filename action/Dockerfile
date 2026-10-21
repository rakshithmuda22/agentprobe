FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY agentprobe/ agentprobe/
COPY .agentprobe/ .agentprobe/

RUN pip install --no-cache-dir . && \
    rm -rf /root/.cache/pip

ENTRYPOINT ["python", "-m", "agentprobe.integrations.action_runner"]
