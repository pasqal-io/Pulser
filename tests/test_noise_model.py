# Copyright 2024 Pulser Development Team
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
import numpy as np
import pytest

from pulser.noise_model import NoiseModel


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

    @pytest.mark.parametrize("value", [-1e-9, 0.2, 1.0001])
    @pytest.mark.parametrize(
        "param",
        [
            "dephasing_rate",
            "hyperfine_dephasing_rate",
            "relaxation_rate",
            "depolarizing_rate",
        ],
    )
    def test_init_rate_like(self, param, value):
        if value < 0:
            with pytest.raises(
                ValueError,
                match=f"'{param}' must be None or greater "
                f"than or equal to zero, not {value}.",
            ):
                NoiseModel(**{param: value})
        else:
            noise_model = NoiseModel(**{param: value})
            assert getattr(noise_model, param) == value

    @pytest.mark.parametrize("value", [-1e-9, 1.0001])
    @pytest.mark.parametrize(
        "param",
        [
            "state_prep_error",
            "p_false_pos",
            "p_false_neg",
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

    @pytest.fixture
    def matrices(self):
        matrices = {}
        matrices["I"] = np.eye(2)
        matrices["X"] = np.ones((2, 2)) - np.eye(2)
        matrices["Y"] = np.array([[0, -1j], [1j, 0]])
        matrices["Zh"] = 0.5 * np.array([[1, 0], [0, -1]])
        matrices["ket"] = np.array([[1.0], [2.0]])
        matrices["I3"] = np.eye(3)
        return matrices

    def test_eff_noise_rates(self, matrices):
        with pytest.raises(
            ValueError, match="The provided rates must be greater than 0."
        ):
            NoiseModel(
                noise_types=("eff_noise",),
                eff_noise_opers=[matrices["I"], matrices["X"]],
                eff_noise_rates=[-1.0, 0.5],
            )

    def test_eff_noise_opers(self, matrices):
        with pytest.raises(ValueError, match="The operators list length"):
            NoiseModel(noise_types=("eff_noise",), eff_noise_rates=[1.0])
        with pytest.raises(
            TypeError, match="eff_noise_rates is a list of floats"
        ):
            NoiseModel(
                noise_types=("eff_noise",),
                eff_noise_rates=["0.1"],
                eff_noise_opers=[np.eye(2)],
            )
        with pytest.raises(
            ValueError,
            match="The effective noise parameters have not been filled.",
        ):
            NoiseModel(noise_types=("eff_noise",))
        with pytest.raises(TypeError, match="not castable to a Numpy array"):
            NoiseModel(
                noise_types=("eff_noise",),
                eff_noise_rates=[2.0],
                eff_noise_opers=[{(1.0, 0), (0.0, -1)}],
            )
        with pytest.raises(ValueError, match="is not a 2D array."):
            NoiseModel(
                noise_types=("eff_noise",),
                eff_noise_opers=[2.0],
                eff_noise_rates=[1.0],
            )
        with pytest.raises(NotImplementedError, match="Operator's shape"):
            NoiseModel(
                noise_types=("eff_noise",),
                eff_noise_opers=[matrices["I3"]],
                eff_noise_rates=[1.0],
            )

    def test_eq(self, matrices):
        final_fields = dict(
            noise_types=("SPAM", "eff_noise"),
            eff_noise_rates=(0.1, 0.4),
            eff_noise_opers=(((0, 1), (1, 0)), ((0, -1j), (1j, 0))),
        )
        noise_model = NoiseModel(
            noise_types=["SPAM", "eff_noise"],
            eff_noise_rates=[0.1, 0.4],
            eff_noise_opers=[matrices["X"], matrices["Y"]],
        )
        assert noise_model == NoiseModel(**final_fields)
        for param in final_fields:
            assert final_fields[param] == getattr(noise_model, param)
