"""Definition of a noise trajectory."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class NoiseTrajectory:
    """Defines a noise trajectory."""
    bad_atoms: dict[str, bool] = field(default_factory=dict)
    doppler_detune: dict[str, float] = field(default_factory=dict)
    amp_fluctuations: dict[str, float] = field(default_factory=dict)
    det_fluctuations: dict[str, float] = field(default_factory=dict)
    det_phases: dict[str, np.ndarray] = field(default_factory=dict)
