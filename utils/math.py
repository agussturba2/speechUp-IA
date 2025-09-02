"""Mathematical helper functions used across the codebase."""

from numba import jit


@jit(nopython=False, forceobj=True)
def calculate_posture_score(
    l_shoulder_x: float, r_shoulder_x: float, nose_x: float
) -> bool:
    """Return True if head is roughly centered between shoulders.

    Using Numba JIT for speed because this function is called on every frame.
    The threshold (0.08) comes from empirical testing on 0–1 normalized coords.
    """
    return abs((l_shoulder_x + r_shoulder_x) / 2 - nose_x) < 0.08
