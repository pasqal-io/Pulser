"""Contains noise models.

For now, only the amplitude and doppler noises are implemented in the form a
NoiseModel, which are the laser-atom interaction related noises relevant at
sampling time.
"""
from __future__ import annotations

import functools
from typing import Callable, Optional

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
    reg: Register,
    waist_width: float,
    random: bool = True,
    seed: Optional[int] = None,
) -> NoiseModel:
    """Generate a NoiseModel for the gaussian amplitude profile of laser beams.

    The laser of a global channel has a non-constant amplitude profile in the
    register plane. It makes global channels act differently on each qubit,
    becoming local.

    Args:
        reg (Register): A Pulser register
        waist_width (float): The laser waist_width in µm
        random (bool): Adds an additional random noise on the amplitude
        seed (int): Optional, seed for the numpy.random.Generator

    Return:
        NoiseModel: The function that applies the amplitude noise to some
        QubitSamples.
    """
    rng = np.random.default_rng(seed)

    def f(s: QubitSamples) -> QubitSamples:
        r = np.linalg.norm(reg.qubits[s.qubit])

        noise_amp = rng.normal(1.0, 1.0e-3) if random else 1.0
        noise_amp *= np.exp(-((r / waist_width) ** 2))

        amp = s.amp.copy()
        amp *= noise_amp

        return QubitSamples(
            amp=amp,
            det=s.det.copy(),
            phase=s.phase.copy(),
            qubit=s.qubit,
        )

    return f


def doppler(reg: Register, std_dev: float, seed: Optional[int]) -> NoiseModel:
    """Generate a NoiseModel for the Doppler effect detuning shifts.

    Example usage:

        MASS = 1.45e-25  # kg
        KB = 1.38e-23  # J/K
        KEFF = 8.7  # µm^-1
        sigma = KEFF * np.sqrt(KB * 50.0e-6 / MASS)
        doppler_noise = doppler(reg, sigma)
        ...

    Args:
        reg (Register): A Pulser register
        std_dev (float): The standard deviation of the normal distribution used
            to sample the random detuning shifts
        seed (int): Optional, seed for the numpy.random.Generator

    Return:
        NoiseModel: The function that applies the doppler noise to some
        QubitSamples.
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
    """Helper to compose multiple NoiseModel.

    Args:
        *functions: a list of functions

    Returns:
        The mathematical composition of *functions. The last element is applied
        first. If *functions is [f, g, h], it returns f∘g∘h.
    """
    return functools.reduce(
        lambda f, g: lambda x: f(g(x)), functions, lambda x: x
    )


def apply(
    samples: list[QubitSamples], noises: list[NoiseModel]
) -> list[QubitSamples]:
    """Apply a list of NoiseModel on a list of QubitSamples.

    The noises are composed using the compose_local_noises function, such that
    the last element is applied first.

    Args:
        samples (list[QubitSamples]): A list of QubitSamples.
        noises (list[NoiseModel]): A list of NoiseModel.

    Return:
        A list of QubitSamples on which each element of noises has been
        applied.
    """
    tot_noise = compose_local_noises(*noises)

    return [tot_noise(s) for s in samples]
