"""Defines samples dataclasses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulser.sequence import QubitId


@dataclass
class Samples:
    """Gather samples for unspecified qubits."""

    amp: np.ndarray
    det: np.ndarray
    phase: np.ndarray


@dataclass
class QubitSamples:
    """Gathers samples concerning a single qubit."""

    amp: np.ndarray
    det: np.ndarray
    phase: np.ndarray
    qubit: QubitId

    @classmethod
    def from_global(cls, qubit: QubitId, s: Samples) -> QubitSamples:
        """Construct a QubitSamples from a Samples instance."""
        return cls(amp=s.amp, det=s.det, phase=s.phase, qubit=qubit)

    def __post_init__(self) -> None:
        if not len(self.amp) == len(self.det) == len(self.phase):
            raise ValueError(
                "ndarrays amp, det and phase must have the same length"
            )
