"""Small, dependency-free analysis functions based on the CEP manuscript."""
import math
from statistics import fmean


def tracking_decomposition(forward_velocities, lateral_velocities, command):
    """Return planar RMSE² = forward bias² + lateral bias² + velocity ripple²."""
    vx, vy = list(forward_velocities), list(lateral_velocities)
    if not vx or len(vx) != len(vy):
        raise ValueError("velocity sequences must be nonempty and equally sized")
    if not math.isfinite(command) or not all(math.isfinite(v) for v in vx + vy):
        raise ValueError("velocities and command must be finite")
    mean_x = fmean(vx)
    bias_x = mean_x - command
    bias_y = fmean(vy)
    ripple_squared = fmean((x - mean_x) ** 2 + (y - bias_y) ** 2 for x, y in zip(vx, vy))
    rmse_squared = fmean((x - command) ** 2 + y ** 2 for x, y in zip(vx, vy))
    return {"forward_bias": bias_x, "lateral_bias": bias_y,
            "ripple": math.sqrt(ripple_squared), "planar_rmse": math.sqrt(rmse_squared)}


def constant_hazard(deaths, exposure_s):
    """Return deaths divided by exposure after the selected transient."""
    if isinstance(deaths, bool) or not isinstance(deaths, int) or deaths < 0:
        raise ValueError("deaths must be a nonnegative integer")
    if not math.isfinite(exposure_s) or exposure_s <= 0:
        raise ValueError("exposure must be positive and finite")
    return deaths / exposure_s


def conditional_survival(hazard, elapsed_s):
    if not math.isfinite(hazard) or not math.isfinite(elapsed_s) or hazard < 0 or elapsed_s < 0:
        raise ValueError("hazard and elapsed time must be nonnegative and finite")
    return math.exp(-hazard * elapsed_s)
