"""Shared test fixtures.

Tests that need the real competition files are marked `data` and are skipped
automatically when the files are not present.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from cairn.data import TASK_FILES, TaskData, data_dir, load_task
from cairn.tokens import HeuristicCounter, TokenCounter, get_counter
from cairn.types import Example

HAVE_DATA = all((data_dir() / name).exists() for name in TASK_FILES.values())


def _triton_python() -> str | None:
    """An interpreter that can import torch and triton, if this machine has one."""
    candidates = [os.environ.get("CAIRN_TRITON_PYTHON"), sys.executable]
    for exe in filter(None, candidates):
        if shutil.which(exe) or Path(exe).exists():
            import subprocess

            probe = subprocess.run([exe, "-c", "import torch, triton"], capture_output=True)
            if probe.returncode == 0:
                return exe
    return None


TRITON_PYTHON = _triton_python()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "data" in item.keywords and not HAVE_DATA:
            item.add_marker(pytest.mark.skip(reason="competition data files not found"))
        if "triton" in item.keywords and TRITON_PYTHON is None:
            item.add_marker(
                pytest.mark.skip(reason="torch + triton not installed (set CAIRN_TRITON_PYTHON)")
            )


@pytest.fixture(scope="session")
def counter() -> TokenCounter:
    return get_counter()


@pytest.fixture(scope="session")
def heuristic_counter() -> TokenCounter:
    return HeuristicCounter()


_TASK_CACHE: dict[int, TaskData] = {}


@pytest.fixture(scope="session")
def task_loader():
    def load(task_id: int) -> TaskData:
        if task_id not in _TASK_CACHE:
            _TASK_CACHE[task_id] = load_task(task_id)
        return _TASK_CACHE[task_id]

    return load


@pytest.fixture(scope="session")
def triton_python() -> str:
    assert TRITON_PYTHON is not None
    return TRITON_PYTHON


def make_sentiment_pool(n: int = 400) -> list[Example]:
    """A small synthetic two-label pool, big enough to exercise every code path fast."""
    happy = ["great", "love", "wonderful", "sunny", "smile", "win", "fun", "delight"]
    sad = ["awful", "cry", "lonely", "rain", "tears", "lost", "grief", "hurt"]
    filler = ["today", "the", "really", "so", "my", "day", "was", "feeling", "and", "with"]
    pool = []
    for i in range(n):
        positive = i % 3 != 0  # 2/3 positive: an unbalanced pool on purpose
        words = happy if positive else sad
        text = " ".join(
            [
                filler[i % 10],
                words[i % 8],
                filler[(i * 7) % 10],
                words[(i * 3) % 8],
                filler[(i * 3) % 10],
                f"t{i}",
            ]
        )
        pool.append(Example(f"ex{i}", text, "Not sad" if positive else "Sad"))
    return pool


@pytest.fixture()
def sentiment_pool() -> list[Example]:
    return make_sentiment_pool()
