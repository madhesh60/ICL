"""Cairn: a long-context in-context-learning (ICL) annotation engine for small LLMs.

The one idea behind the whole package:

    The pool of labelled examples is much bigger than the context window.
    The examples that do not fit in the prompt are not waste.
    They are a free test set, a free calibration set and a free tuning set.

Everything else (what goes in the prompt, where it goes, how the answer is read,
when to spend more compute) is decided by measuring against those left-over examples.
"""

from cairn.types import Example, Prediction, Query

__version__ = "0.1.0"
__all__ = ["Example", "Prediction", "Query", "__version__"]
