"""
Non-lexicase selection methods, for comparison in examples and benchmarks.

These are deliberately not part of the `lexicase` package. They exist so the
examples can show what lexicase selection is being compared against, and they
are not covered by the package's API stability.
"""

from .fitness_proportionate import fitness_proportionate_selection
from .tournament import tournament_selection

__all__ = ["fitness_proportionate_selection", "tournament_selection"]
