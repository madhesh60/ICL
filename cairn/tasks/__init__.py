"""What is special about each of the 8 tasks, written down as data."""

from cairn.tasks.base import TaskSpec, View
from cairn.tasks.registry import TASKS, get_task

__all__ = ["TASKS", "TaskSpec", "View", "get_task"]
