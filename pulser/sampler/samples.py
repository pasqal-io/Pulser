"""Defines samples dataclasses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulser.sequence import QubitId


@dataclass
class GlobalSamples:
    """Samples gather arrays of values for amplitude, detuning and phase."""

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
    def from_global(cls, qubit: QubitId, s: GlobalSamples) -> QubitSamples:
        """Construct a QubitSamples from a Samples instance."""
        return cls(amp=s.amp, det=s.det, phase=s.phase, qubit=qubit)
