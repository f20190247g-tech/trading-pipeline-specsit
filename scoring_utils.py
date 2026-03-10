"""
Shared Scoring Utilities — normalization, grading, percentile ranking.
Used across all phases: conviction scoring, value analysis, macro liquidity, etc.
"""


def normalize_score(value: float, min_val: float, max_val: float, invert: bool = False) -> float:
    """Normalize a value to 0-100 scale. Clamps to bounds.

    Args:
        value: Raw value to normalize.
        min_val: Value that maps to 0 (or 100 if inverted).
        max_val: Value that maps to 100 (or 0 if inverted).
        invert: If True, higher raw value = lower score.
    """
    if max_val == min_val:
        return 50.0
    score = (value - min_val) / (max_val - min_val) * 100
    score = max(0.0, min(100.0, score))
    if invert:
        score = 100.0 - score
    return round(score, 1)


def grade_score(score: float, thresholds: dict | None = None) -> str:
    """Map a 0-100 score to a letter grade.

    Args:
        score: 0-100 score.
        thresholds: Optional custom thresholds. Default:
            A+ >= 90, A >= 75, B >= 60, C >= 40, D < 40
    """
    if thresholds is None:
        thresholds = {"A+": 90, "A": 75, "B": 60, "C": 40}

    for grade, cutoff in sorted(thresholds.items(), key=lambda x: -x[1]):
        if score >= cutoff:
            return grade
    return "D"


def weighted_composite(components: list[tuple[float, float]]) -> float:
    """Compute weighted average from (score, weight) tuples.

    Args:
        components: List of (score, weight) tuples. Weights are normalized internally.

    Returns:
        Weighted average score 0-100.
    """
    total_weight = sum(w for _, w in components if w > 0)
    if total_weight == 0:
        return 0.0
    result = sum(s * w for s, w in components) / total_weight
    return round(max(0.0, min(100.0, result)), 1)


def percentile_rank(value: float, distribution: list[float]) -> float:
    """Compute percentile rank of value within distribution (0-100).

    Args:
        value: The value to rank.
        distribution: All values in the population.

    Returns:
        Percentile 0-100.
    """
    if not distribution:
        return 50.0
    count_below = sum(1 for v in distribution if v <= value)
    return round(count_below / len(distribution) * 100, 1)
