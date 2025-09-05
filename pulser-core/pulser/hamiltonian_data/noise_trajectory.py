"""Definition of a noise trajectory."""

from dataclasses import dataclass, field

import numpy as np
from pulser.register.base_register import BaseRegister
from pulser import Register


@dataclass(frozen=True)
class NoiseTrajectory:
    """Defines a noise trajectory."""

    bad_atoms: dict[str, bool] = field(default_factory=dict)
    doppler_detune: dict[str, float] = field(default_factory=dict)
    amp_fluctuations: dict[str, float] = field(default_factory=dict)
    det_fluctuations: dict[str, float] = field(default_factory=dict)
    det_phases: dict[str, np.ndarray] = field(default_factory=dict)
    register: BaseRegister = field(default_factory=Register)
