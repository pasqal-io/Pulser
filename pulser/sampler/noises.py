"""Contains the noise models."""
from __future__ import annotations

import functools
from typing import Callable

import numpy as np

from pulser.register import Register
from pulser.sampler.samples import QubitSamples

NoiseModel = Callable[[QubitSamples], QubitSamples]
"""A function that apply some noise on a list of QubitSamples.

A NoiseModel corresponds to a source of noises present in a device which is
relevant when sampling the input pulses. Physical effects contributing to
modifications of the shined amplitude, detuning and phase felt by qubits of the
register are susceptible to be implemented by a NoiseModel.
"""


def amplitude(reg: Register, waist_width: float, seed: int = 0) -> NoiseModel:
    """Generate a NoiseModel for the gaussian amplitude profile of laser beams.

    The laser of a global channel has a non-constant amplitude profile in the
    register plane. It makes global channels act differently on each qubit,
    becoming local.
    """
    rng = np.random.default_rng(seed)

    def f(s: QubitSamples) -> QubitSamples:
        r = np.linalg.norm(reg.qubits[s.qubit])

        noise_amp = rng.normal(1.0, 1.0e-3)
        noise_amp *= np.exp(-((r / waist_width) ** 2))

        amp = s.amp
        amp[np.nonzero(s.amp)] *= noise_amp
        return QubitSamples(
            amp=s.amp,
            det=s.det,
            phase=s.phase,
            qubit=s.qubit,
        )

    return f


def doppler(reg: Register, std_dev: float, seed: int = 0) -> NoiseModel:
    """Generate a NoiseModel for the Doppler effect detuning shifts."""
    rng = np.random.default_rng(seed)
    errs = rng.normal(0.0, std_dev, size=len(reg.qubit_ids))
    detunings = dict(zip(reg.qubit_ids, errs))

    def f(s: QubitSamples) -> QubitSamples:
        det = s.det
        det[np.nonzero(s.det)] += detunings[s.qubit]
        return QubitSamples(
            amp=s.amp,
            det=det,
            phase=s.phase,
            qubit=s.qubit,
        )

    return f


def compose_local_noises(*functions: NoiseModel) -> NoiseModel:
    """Helper to compose multiple NoiseModel."""
    if functions is None:
        return lambda x: x
    return functools.reduce(
        lambda f, g: lambda x: f(g(x)), functions, lambda x: x
    )


def apply(
    samples: list[QubitSamples], noises: list[NoiseModel]
) -> list[QubitSamples]:
    """Apply a list of NoiseModel on a list of QubitSamples."""
    tot_noise = compose_local_noises(*noises)

    return [tot_noise(s) for s in samples]
