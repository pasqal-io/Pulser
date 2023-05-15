# Copyright 2023 Pulser Development Team
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

import pytest
import numpy as np

import pulser
from pulser.devices import MockDevice
from pulser.backend.abc import Backend
from pulser.backend.noise_model import NoiseModel


@pytest.fixture
def sequence() -> pulser.Sequence:
    reg = pulser.Register.square(2, spacing=5, prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.add(pulser.Pulse.ConstantPulse(1000, 1, -1, 0), "rydberg_global")
    return seq


def test_abc_backend(sequence):
    with pytest.raises(
        TypeError, match="Can't instantiate abstract class Backend"
    ):
        Backend(sequence)

    class ConcreteBackend(Backend):
        def run(self):
            pass

    with pytest.raises(
        TypeError, match="'sequence' should be a `Sequence` instance"
    ):
        ConcreteBackend(sequence.to_abstract_repr())


class TestNoiseModel:
    def test_bad_noise_type(self):
        with pytest.raises(
            ValueError, match="'bad_noise' is not a valid noise type."
        ):
            NoiseModel(noise_types=("bad_noise",))

    @pytest.mark.parametrize(
        "param",
        ["runs", "samples_per_run", "temperature", "laser_waist"],
    )
    def test_init_strict_pos(self, param):
        with pytest.raises(
            ValueError, match=f"'{param}' must be greater than zero, not 0"
        ):
            NoiseModel(**{param: 0})

    @pytest.mark.parametrize("value", [-1e-9, 1.0001])
    @pytest.mark.parametrize(
        "param",
        [
            "state_prep_error",
            "p_false_pos",
            "p_false_neg",
            "dephasing_prob",
            "depolarizing_prob",
            "amp_sigma",
        ],
    )
    def test_init_prob_like(self, param, value):
        with pytest.raises(
            ValueError,
            match=f"'{param}' must be greater than or equal to zero and "
            f"smaller than or equal to one, not {value}",
        ):
            NoiseModel(**{param: value})

    @pytest.mark.parametrize(
        "noise_sample,",
        [
            ("dephasing", "depolarizing"),
            ("eff_noise", "depolarizing"),
            ("eff_noise", "dephasing"),
            ("depolarizing", "eff_noise", "dephasing"),
        ],
    )
    def test_eff_noise_init(self, noise_sample):
        with pytest.raises(
            NotImplementedError,
            match="Depolarizing, dephasing and effective noise channels",
        ):
            NoiseModel(noise_types=noise_sample)

    @pytest.fixture
    def matrices(self):
        matrices = {}
        matrices["I"] = np.eye(2)
        matrices["X"] = np.ones((2, 2)) - np.eye(2)
        matrices["Zh"] = 0.5 * np.array([[1, 0], [0, -1]])
        matrices["ket"] = np.array([[1.0], [2.0]])
        matrices["I3"] = np.eye(3)
        return matrices

    @pytest.mark.parametrize(
        "prob_distr",
        [
            [-1.0, 0.5],
            [0.5, 2.0],
            [0.3, 0.2],
        ],
    )
    def test_eff_noise_probs(self, prob_distr, matrices):
        with pytest.raises(
            ValueError, match="is not a probability distribution."
        ):
            NoiseModel(
                noise_types=("eff_noise",),
                eff_noise_opers=[matrices["I"], matrices["X"]],
                eff_noise_probs=prob_distr,
            )

    def test_eff_noise_opers(self, matrices):
        with pytest.raises(ValueError, match="The operators list length"):
            NoiseModel(noise_types=("eff_noise",), eff_noise_probs=[1.0])
        with pytest.raises(
            TypeError, match="eff_noise_probs is a list of floats"
        ):
            NoiseModel(
                noise_types=("eff_noise",),
                eff_noise_probs=["0.1"],
                eff_noise_opers=[np.eye(2)],
            )
        with pytest.raises(
            ValueError,
            match="The general noise parameters have not been filled.",
        ):
            NoiseModel(noise_types=("eff_noise",))
        with pytest.raises(TypeError, match="is not a Numpy array."):
            NoiseModel(
                noise_types=("eff_noise",),
                eff_noise_opers=[2.0],
                eff_noise_probs=[1.0],
            )
        with pytest.raises(NotImplementedError, match="Operator's shape"):
            NoiseModel(
                noise_types=("eff_noise",),
                eff_noise_opers=[matrices["I3"]],
                eff_noise_probs=[1.0],
            )
        with pytest.raises(
            NotImplementedError, match="You must put the identity matrix"
        ):
            NoiseModel(
                noise_types=("eff_noise",),
                eff_noise_opers=[matrices["X"], matrices["I"]],
                eff_noise_probs=[0.5, 0.5],
            )
        with pytest.raises(
            ValueError, match="The completeness relation is not"
        ):
            NoiseModel(
                noise_types=("eff_noise",),
                eff_noise_opers=[matrices["I"], matrices["Zh"]],
                eff_noise_probs=[0.5, 0.5],
            )
