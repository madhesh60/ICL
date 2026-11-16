"""Reading settings from a YAML file into typed config objects.

One rule matters most here: an unknown key is an error, not something to ignore.
If you write `tail_kk: 20` by mistake and the loader silently drops it, you will
run a whole experiment with the old value and never know.  Failing loudly at
start-up costs a second; a silently wrong experiment costs a day.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from cairn.context.builder import ContextConfig
from cairn.data import SplitSizes
from cairn.pipeline.annotator import StrategyConfig
from cairn.pipeline.runner import RunConfig

T = TypeVar("T")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BackendConfig:
    kind: str = "openai"  # "openai" (vLLM / FlagScale / any OpenAI-style server) or "demo-reader"
    base_url: str = "http://127.0.0.1:2026/v1"
    model: str = "Qwen3-4B"
    api_key: str = "EMPTY"
    timeout_s: float = 600.0
    structured_api: str = "structured_outputs"  # or "guided" (old vLLM) or "none"
    thinking_control: bool = True
    max_attempts: int = 5
    response_cache: bool = True


@dataclass(frozen=True)
class AppConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    defaults: dict[str, Any] = field(default_factory=dict)
    tasks: dict[int, dict[str, Any]] = field(default_factory=dict)

    def for_task(self, task_id: int) -> RunConfig:
        merged = _deep_merge(self.defaults, self.tasks.get(task_id, {}))
        return _run_config(merged)


def _build(cls: type[T], values: dict[str, Any], where: str) -> T:
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(values) - names)
    if unknown:
        raise ConfigError(
            f"unknown setting(s) in {where}: {', '.join(unknown)}. Allowed: {', '.join(sorted(names))}"
        )
    return cls(**values)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _run_config(values: dict[str, Any]) -> RunConfig:
    values = dict(values)
    parts = {
        "context": _build(ContextConfig, values.pop("context", {}) or {}, "context"),
        "strategy": _build(StrategyConfig, values.pop("strategy", {}) or {}, "strategy"),
        "splits": _build(SplitSizes, values.pop("splits", {}) or {}, "splits"),
    }
    return _build(RunConfig, {**values, **parts}, "run settings")


def load_config(path: Path | str | None) -> AppConfig:
    """Load a YAML file; with no path, return the built-in defaults."""
    if path is None:
        return AppConfig()
    file = Path(path)
    if not file.exists():
        raise ConfigError(f"config file not found: {file}")
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{file} must contain a mapping at the top level")
    unknown = sorted(set(raw) - {"backend", "defaults", "tasks"})
    if unknown:
        raise ConfigError(f"unknown top-level section(s) in {file}: {', '.join(unknown)}")
    tasks: dict[int, dict[str, Any]] = {}
    for key, value in (raw.get("tasks") or {}).items():
        try:
            task_id = int(key)
        except (TypeError, ValueError) as err:
            raise ConfigError(f"task keys must be numbers 1..8, got {key!r}") from err
        if task_id not in range(1, 9):
            raise ConfigError(f"task keys must be numbers 1..8, got {key!r}")
        tasks[task_id] = value or {}
    config = AppConfig(
        backend=_build(BackendConfig, raw.get("backend") or {}, "backend"),
        defaults=raw.get("defaults") or {},
        tasks=tasks,
    )
    # Build every task's settings once now, so typos surface at start-up
    # and not three hours into a run.
    for task_id in range(1, 9):
        config.for_task(task_id)
    return config
