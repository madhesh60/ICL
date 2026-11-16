"""Choosing which examples earn a place in the prompt."""

from cairn.selection.submodular import CoverageResult, budgeted_coverage, coverage_value

__all__ = ["CoverageResult", "budgeted_coverage", "coverage_value"]
