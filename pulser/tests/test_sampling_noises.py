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
from __future__ import annotations

import numpy as np
import pytest

import pulser
import pulser.simulation.noises as noises
from pulser.devices import MockDevice
from pulser.pulse import Pulse
from pulser.sampler import sample
from pulser.waveforms import ConstantWaveform


def test_amplitude_noise():
    """Test the noise related to the amplitude profile of global pulses."""
    N = 100
    amplitude = 1.0
    waist_width = 2.0  # µm

    coords = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    reg = pulser.Register.from_coordinates(coords, prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    seq.add(
        Pulse.ConstantAmplitude(amplitude, ConstantWaveform(N, 0.0), 0.0),
        "ch0",
    )
    seq.measure()

    def expected_samples(vec: np.ndarray) -> np.ndarray:
        """Defines the non-noisy effect of a gaussian amplitude profile."""
        r = np.linalg.norm(vec)
        a = np.ones(N)
        a *= amplitude
        a *= np.exp(-((r / waist_width) ** 2))
        return a

    s = sample(
        seq, global_noises=[noises.amplitude(reg, waist_width, random=False)]
    )

    for q, coords in reg.qubits.items():
        got = s["Local"]["ground-rydberg"][q]["amp"]
        want = expected_samples(coords)
        np.testing.assert_equal(got, want)


@pytest.mark.xfail(
    reason="Test a different doppler effect than the one implemented; "
    "no surprise it fails."
)
def test_doppler_noise():
    """What is exactly the doppler noise here?

    A constant detuning shift per pulse seems weird. A global shift seems more
    reasonable, but how can it be constant during the all sequence? It is not
    clear to me here, I find the current implementation in the simulation
    module to be unsatisfactory.

    No surprise I make it fail on purpose right now 😅
    """
    N = 100
    det_value = np.pi

    reg = pulser.Register.from_coordinates(np.array([[0.0, 0.0]]), prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    for _ in range(3):
        seq.add(
            Pulse.ConstantDetuning(ConstantWaveform(N, 1.0), det_value, 0.0),
            "ch0",
        )
        seq.delay(100, "ch0")
    seq.measure()

    MASS = 1.45e-25  # kg
    KB = 1.38e-23  # J/K
    KEFF = 8.7  # µm^-1
    doppler_sigma = KEFF * np.sqrt(KB * 50.0e-6 / MASS)
    seed = 42
    rng = np.random.default_rng(seed)

    shifts = rng.normal(0, doppler_sigma, 3)
    want = np.zeros(6 * N)
    want[0:100] = det_value + shifts[0]
    want[200:300] = det_value + shifts[1]
    want[400:500] = det_value + shifts[2]

    local_noises = [noises.doppler(reg, doppler_sigma, seed=seed)]
    samples = sample(seq, common_noises=local_noises)
    got = samples["Local"]["ground-rydberg"]["q0"]["det"]

    np.testing.assert_array_equal(got, want)
