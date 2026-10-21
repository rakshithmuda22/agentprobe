FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/agentprobe
COPY pyproject.toml README.md LICENSE ./
COPY agentprobe/ agentprobe/
RUN pip install . && rm -rf /root/.cache

# GitHub runs Docker actions with --workdir /github/workspace (the checked-out
# repo). The console script does not put the workspace on sys.path, so code
# in the user's repo can never shadow AgentProbe's own modules.
ENTRYPOINT ["agentprobe", "action"]
