"""Configuration loader for AgentProbe."""

from __future__ import annotations

import dataclasses
import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = ".agentprobe"


@dataclass
class RegressionConfig:
    # Layer 2 (differential property tests) only runs for files under these
    # directories. Empty list = every Python file in the diff.
    critical_dirs: list[str] = field(default_factory=lambda: ["src/payments", "src/auth"])
    # Cap on functions compared per PR (keeps runtime bounded on huge PRs).
    max_functions_per_pr: int = 20
    # Layer 2 executes PR code in a subprocess, so it is opt-in.
    property_tests: bool = False
    property_test_examples: int = 200
    property_test_timeout_seconds: int = 60


@dataclass
class CacheConfig:
    backend: str = "memory"
    ttl_seconds: int = 86400


@dataclass
class LLMConfig:
    # "none" (default) = fully deterministic. "ollama" = ask a local Ollama
    # model for function summaries in Layer 1 (experimental, non-deterministic).
    provider: str = "none"
    model: str = "llama3"
    base_url: str = "http://localhost:11434"


@dataclass
class AgentProbeConfig:
    # Verdict: BLOCK if any FATAL finding or score > block_threshold,
    # WARN if score > warn_threshold, otherwise PASS.
    block_threshold: float = 70
    warn_threshold: float = 0
    weights: dict[str, float] = field(
        default_factory=lambda: {"architecture": 0.40, "pattern": 0.25, "regression": 0.35}
    )
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    # Glob patterns (fnmatch on repo-relative paths) the analyzers skip,
    # e.g. ["examples/*", "tests/fixtures/*", "*_pb2.py"].
    ignore_paths: list[str] = field(default_factory=list)

    def is_ignored(self, path: str) -> bool:
        return any(fnmatch.fnmatch(path, pattern) for pattern in self.ignore_paths)


# Old key names that are still accepted so existing config files keep loading.
_RENAMED_KEYS = {"max_llm_calls_per_pr": "max_functions_per_pr"}


def _build(cls: type, raw: Mapping[str, Any] | None, section: str):
    """Instantiate a config dataclass, ignoring (and logging) unknown keys."""
    raw = dict(raw or {})
    known = {f.name for f in dataclasses.fields(cls)}
    for old, new in _RENAMED_KEYS.items():
        if old in raw and new in known and new not in raw:
            raw[new] = raw.pop(old)
    unknown = sorted(set(raw) - known)
    if unknown:
        logger.warning("Ignoring unknown keys in config.yaml [%s]: %s", section, unknown)
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(config_path: str | Path = ".agentprobe/config.yaml") -> AgentProbeConfig:
    """Load AgentProbe config from a YAML file. Missing file -> defaults."""
    path = Path(config_path)
    if not path.exists():
        return AgentProbeConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    defaults = AgentProbeConfig()
    thresholds = raw.get("thresholds") or {}
    weights = dict(defaults.weights)
    weights.update(raw.get("weights") or {})
    return AgentProbeConfig(
        block_threshold=thresholds.get("block", defaults.block_threshold),
        warn_threshold=thresholds.get("warn", defaults.warn_threshold),
        weights=weights,
        regression=_build(RegressionConfig, raw.get("regression"), "regression"),
        cache=_build(CacheConfig, raw.get("cache"), "cache"),
        llm=_build(LLMConfig, raw.get("llm"), "llm"),
        ignore_paths=[str(p) for p in raw.get("ignore_paths") or []],
    )


def policy_dir(state: Mapping[str, Any]) -> Path:
    """Directory holding config.yaml / boundaries.yaml / style-profile.yaml.

    The action and CLI point this at a copy of the *base* branch's policy so a
    PR cannot relax the rules it is being judged by. Defaults to
    ``<repo_path>/.agentprobe``.
    """
    explicit = state.get("config_dir")
    if explicit:
        return Path(explicit)
    return Path(state.get("repo_path") or ".") / DEFAULT_CONFIG_DIR
