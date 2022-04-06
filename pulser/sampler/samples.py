"""Defines samples dataclasses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulser.sequence import QubitId


@dataclass
class QubitSamples:
    """Gathers samples concerning a single qubit."""

    amp: np.ndarray
    det: np.ndarray
    phase: np.ndarray
    qubit: QubitId

    def __post_init__(self) -> None:
        if not len(self.amp) == len(self.det) == len(self.phase):
            raise ValueError(
                "ndarrays amp, det and phase must have the same length."
            )
