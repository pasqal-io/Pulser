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
"""Defines a NoiseModel and how to apply it to the samples."""
from __future__ import annotations

import functools
from typing import Callable

from pulser.sampler.samples import QubitSamples

NoiseModel = Callable[[QubitSamples], QubitSamples]
"""A function that apply some noise on a list of QubitSamples.

A NoiseModel corresponds to a source of noises present in a device which is
relevant when sampling the input pulses. Physical effects contributing to
modifications of the shined amplitude, detuning and phase felt by qubits of the
register are susceptible to be implemented by a NoiseModel.
"""


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


def apply_noises(
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
