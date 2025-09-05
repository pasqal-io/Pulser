"""Definition of a noise trajectory."""

from dataclasses import dataclass, field

import numpy as np

from pulser.register.base_register import BaseRegister


@dataclass(frozen=True)
class NoiseTrajectory:
    """Defines a noise trajectory."""

    bad_atoms: dict[str, bool] = field()
    doppler_detune: dict[str, float] = field()
    amp_fluctuations: dict[str, float] = field()
    det_fluctuations: dict[str, float] = field()
    det_phases: dict[str, np.ndarray] = field()
    register: BaseRegister = field()
