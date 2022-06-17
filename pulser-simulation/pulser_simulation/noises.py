# Copyright 2022 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains noise models.

For now, only the amplitude and doppler noises are implemented in the form a
NoiseModel, which are the laser-atom interaction related noises relevant at
sampling time.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pulser.register import Register
from pulser.sampler.noise_model import NoiseModel
from pulser.sampler.samples import QubitSamples


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
        reg: A Pulser register
        waist_width: The laser waist_width in µm
        random: Adds an additional random noise on the amplitude
        seed: seed for the numpy.random.Generator

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
        reg: A Pulser register
        std_dev: The standard deviation of the normal distribution used
            to sample the random detuning shifts
        seed: seed for the numpy.random.Generator

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
