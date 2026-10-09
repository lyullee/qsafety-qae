"""Data-independent geometric schedules shared by the research scripts.

The allocation matches the original study helper for valid inputs. Each shot
at Grover power m costs 2*m+1 equivalent state-preparation queries.
"""

from __future__ import annotations

import math

BASE_SHOTS = 64


def schedule_for_ratio(ratio: float, target_budget: int) -> tuple[list[int], int, int]:
    if not math.isfinite(ratio) or ratio <= 1:
        raise ValueError("ratio must be finite and greater than one")
    if isinstance(target_budget, bool) or not isinstance(target_budget, int) or target_budget < 1:
        raise ValueError("target_budget must be a positive integer")
    calls_limit = max(1, target_budget // BASE_SHOTS)
    candidates = [0, 1]
    while candidates[-1] <= calls_limit:
        next_power = max(
            candidates[-1] + 1,
            math.floor(ratio * candidates[-1] + 0.5),
        )
        candidates.append(next_power)
    selected: list[int] = []
    calls = 0
    for power in candidates:
        proposed = calls + 2 * power + 1
        if proposed > calls_limit:
            break
        selected.append(power)
        calls = proposed
    if not selected:
        selected = [0]
        calls = 1
    shots = max(1, target_budget // calls)
    return selected, shots, shots * calls
