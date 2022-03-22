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


def amplitude(
    reg: Register, waist_width: float, random: bool = True, seed: int = 0
) -> NoiseModel:
    """Generate a NoiseModel for the gaussian amplitude profile of laser beams.

    The laser of a global channel has a non-constant amplitude profile in the
    register plane. It makes global channels act differently on each qubit,
    becoming local.

    Args:
        reg (Register): a Pulser register
        waist_width (float): the laser waist_width in µm
        random (bool): adds an additional random noise on the amplitude
        seed (int): optional, seed for the numpy.random.Generator
    """
    rng = np.random.default_rng(seed)

    def f(s: QubitSamples) -> QubitSamples:
        r = np.linalg.norm(reg.qubits[s.qubit])

        noise_amp = rng.normal(1.0, 1.0e-3) if random else 1.0
        noise_amp *= np.exp(-((r / waist_width) ** 2))

        amp = s.amp.copy()
        amp[np.nonzero(amp)] *= noise_amp

        return QubitSamples(
            amp=amp,
            det=s.det.copy(),
            phase=s.phase.copy(),
            qubit=s.qubit,
        )

    return f


def doppler(reg: Register, std_dev: float, seed: int = 0) -> NoiseModel:
    """Generate a NoiseModel for the Doppler effect detuning shifts.

    Example usage:

        MASS = 1.45e-25  # kg
        KB = 1.38e-23  # J/K
        KEFF = 8.7  # µm^-1
        sigma = KEFF * np.sqrt(KB * 50.0e-6 / MASS)
        doppler_noise = doppler(reg, sigma)
        ...

    Args:
        reg (Register): a Pulser register
        std_dev (float): the standard deviation of the normal distribution used
            to sample the random detuning shifts
        seed (int): optional, seed for the numpy.random.Generator
    """
    rng = np.random.default_rng(seed)
    errs = rng.normal(0.0, std_dev, size=len(reg.qubit_ids))
    detunings = dict(zip(reg.qubit_ids, errs))

    def f(s: QubitSamples) -> QubitSamples:
        det = s.det.copy()
        det[np.nonzero(s.det)] += detunings[s.qubit]
        return QubitSamples(
            amp=s.amp.copy(),
            det=det,
            phase=s.phase.copy(),
            qubit=s.qubit,
        )

    return f


def compose_local_noises(*functions: NoiseModel) -> NoiseModel:
    """Helper to compose multiple NoiseModel."""
    return functools.reduce(
        lambda f, g: lambda x: f(g(x)), functions, lambda x: x
    )


def apply(
    samples: list[QubitSamples], noises: list[NoiseModel]
) -> list[QubitSamples]:
    """Apply a list of NoiseModel on a list of QubitSamples."""
    tot_noise = compose_local_noises(*noises)

    return [tot_noise(s) for s in samples]
