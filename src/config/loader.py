"""Configuration loader for AgentProbe."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RegressionConfig:
    critical_dirs: list[str] = field(default_factory=lambda: ["src/payments", "src/auth"])
    max_llm_calls_per_pr: int = 20
    timing_divergence_threshold: float = 0.20


@dataclass
class CacheConfig:
    backend: str = "memory"
    ttl_seconds: int = 86400


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "llama3"
    base_url: str = "http://localhost:11434"


@dataclass
class AgentProbeConfig:
    block_threshold: int = 70
    warn_threshold: int = 40
    weights: dict[str, float] = field(
        default_factory=lambda: {"architecture": 0.40, "pattern": 0.25, "regression": 0.35}
    )
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def load_config(config_path: str | Path = ".agentprobe/config.yaml") -> AgentProbeConfig:
    """Load AgentProbe config from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        return AgentProbeConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    thresholds = raw.get("thresholds", {})
    return AgentProbeConfig(
        block_threshold=thresholds.get("block", 70),
        warn_threshold=thresholds.get("warn", 40),
        weights=raw.get("weights", {"architecture": 0.40, "pattern": 0.25, "regression": 0.35}),
        regression=RegressionConfig(**raw.get("regression", {})),
        cache=CacheConfig(**raw.get("cache", {})),
        llm=LLMConfig(**raw.get("llm", {})),
    )
