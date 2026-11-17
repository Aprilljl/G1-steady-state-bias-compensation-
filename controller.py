"""Reference implementation of the CEP manuscript's cycle-level control law.

This is a fresh implementation from the manuscript equation, not the original
experiment code. The caller supplies full-cycle mean velocity measurements.
"""
from dataclasses import dataclass
import math


@dataclass
class CycleIntegralController:
    gain: float
    command_min: float
    command_max: float
    activation_time: float = 2.0
    integral: float = 0.0

    def __post_init__(self):
        values = (self.gain, self.command_min, self.command_max, self.activation_time, self.integral)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("controller parameters must be finite")
        if self.gain < 0 or self.activation_time < 0 or self.command_min > self.command_max:
            raise ValueError("invalid gain, activation time, or command bounds")

    def reset(self):
        self.integral = 0.0

    def update(self, reference: float, cycle_mean_velocity: float, time_s: float) -> float:
        """Update at a caller-provided gait-cycle boundary; hold output between calls."""
        if not all(math.isfinite(value) for value in (reference, cycle_mean_velocity, time_s)):
            raise ValueError("reference, velocity and time must be finite")
        if time_s < 0:
            raise ValueError("time must be nonnegative")
        if time_s < self.activation_time:
            return reference
        lower = self.command_min - reference
        upper = self.command_max - reference
        self.integral = min(upper, max(lower, self.integral + self.gain * (reference - cycle_mean_velocity)))
        return reference + self.integral
