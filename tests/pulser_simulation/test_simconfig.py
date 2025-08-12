# Copyright 2020 Pulser Development Team
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

from unittest.mock import patch

import numpy as np
import pytest
from qutip import Qobj, qeye, sigmax, sigmaz

from pulser.noise_model import NoiseModel
from pulser_simulation.simconfig import (
    SimConfig,
    doppler_sigma,
    noisy_register,
    register_sigma_xy_z,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:'SimConfig' has been deprecated:DeprecationWarning"
)


@pytest.fixture
def matrices():
    pauli = {}
    pauli["I"] = qeye(2)
    pauli["X"] = sigmax()
    pauli["Zh"] = 0.5 * sigmaz()
    pauli["ket"] = Qobj([[1.0], [2.0]])
    pauli["I3"] = qeye(3)
    pauli["I4"] = qeye(4)
    return pauli


@pytest.mark.filterwarnings("ignore:Setting samples_per_run different to 1 is")
def test_init():
    with pytest.deprecated_call(match="'SimConfig' has been deprecated"):
        config = SimConfig(
            noise=(
                "SPAM",
                "doppler",
                "dephasing",
                "register",
                "amplitude",
            ),
            temperature=1000.0,
            trap_waist=1.0,
            trap_depth=100.0,
            runs=100,
        )
    expected_temperature = 1000.0
    expected_waist = 1.0
    expected_depth = 100.0
    runs = 100

    assert config.temperature == expected_temperature * 1e-6  # in K
    str_config = config.__str__(True)
    assert "SPAM, doppler, dephasing, register, amplitude" in str_config
    assert (
        f"{expected_temperature}µK" in str_config
        and f"{expected_waist}µm" in str_config
        and f"{expected_depth}µK" in str_config
        and f"{runs}" in str_config
        and "Solver Options" in str_config
    )
    assert config.to_noise_model().temperature == expected_temperature
    assert config.to_noise_model().trap_waist == expected_waist
    assert config.to_noise_model().trap_depth == expected_depth
    assert config.register_sigma_xy_z == register_sigma_xy_z(
        expected_temperature * 1e-6, expected_waist, expected_depth
    )
    config = SimConfig(noise=("depolarizing", "relaxation", "doppler"))
    expected_temperature = 50.0
    assert config.temperature == pytest.approx(
        expected_temperature * 1.0e-6
    )  # 4.9999999999999996e-05
    assert config.to_noise_model().temperature == expected_temperature
    str_config = config.__str__(True)
    assert "depolarizing" in str_config and "relaxation" in str_config
    assert f"Depolarizing rate: {config.depolarizing_rate}" in str_config
    assert f"Relaxation rate: {config.relaxation_rate}" in str_config
    config = SimConfig(
        noise="eff_noise",
        eff_noise_opers=[qeye(2), sigmax()],
        eff_noise_rates=[0.3, 0.7],
    )
    str_config = config.__str__(True)
    assert config.doppler_sigma == doppler_sigma(expected_temperature * 1e-6)
    assert (
        "Effective noise rates" in str_config
        and "Effective noise operators" in str_config
    )

    with pytest.raises(TypeError, match="'temperature' must be a float"):
        SimConfig(temperature="0.0")
    with pytest.raises(ValueError, match="SPAM parameter"):
        SimConfig(eta=-1.0)
    with pytest.raises(
        ValueError, match="'amp_sigma' must be greater than or equal to zero"
    ):
        SimConfig(amp_sigma=-0.001)

    with pytest.raises(
        ValueError, match="'bad_noise' is not a valid noise type."
    ):
        SimConfig(noise=("bad_noise",))


@pytest.mark.filterwarnings("ignore:Setting samples_per_run different to 1 is")
def test_eff_noise_opers(matrices):
    # Some of these checks are repeated in the NoiseModel UTs
    with pytest.raises(ValueError, match="The operators list length"):
        SimConfig(noise=("eff_noise"), eff_noise_rates=[1.0])
    with pytest.raises(TypeError, match="eff_noise_rates is a list of floats"):
        SimConfig(
            noise=("eff_noise"),
            eff_noise_rates=["0.1"],
            eff_noise_opers=[qeye(2)],
        )
    with pytest.raises(
        ValueError,
        match="The effective noise parameters have not been filled.",
    ):
        SimConfig(noise=("eff_noise"))
    with pytest.raises(TypeError, match="is not a Qobj."):
        SimConfig(
            noise=("eff_noise"), eff_noise_opers=[2.0], eff_noise_rates=[1.0]
        )
    with pytest.raises(TypeError, match="to be of Qutip type 'oper'."):
        SimConfig(
            noise=("eff_noise"),
            eff_noise_opers=[matrices["ket"]],
            eff_noise_rates=[1.0],
        )
    with pytest.raises(ValueError, match="With leakage, operator's shape"):
        SimConfig(
            noise=("eff_noise", "leakage"),
            eff_noise_opers=[matrices["I"]],
            eff_noise_rates=[1.0],
        )
    with pytest.raises(ValueError, match="With leakage, operator's shape"):
        SimConfig(
            noise=("eff_noise", "leakage"),
            eff_noise_opers=[qeye(5)],
            eff_noise_rates=[1.0],
        )
    with pytest.raises(ValueError, match="Without leakage, operator's shape"):
        SimConfig(
            noise=("eff_noise",),
            eff_noise_opers=[matrices["I4"]],
            eff_noise_rates=[1.0],
        )
    SimConfig(
        noise=("eff_noise"),
        eff_noise_opers=[matrices["X"], matrices["I"]],
        eff_noise_rates=[0.5, 0.5],
    )


def test_noise_model_conversion():
    noise_model = NoiseModel(
        p_false_neg=0.4,
        p_false_pos=0.1,
        amp_sigma=1e-3,
        runs=10,
        samples_per_run=1,
    )
    expected_simconfig = SimConfig(
        noise=("SPAM", "amplitude"),
        epsilon=0.1,
        epsilon_prime=0.4,
        eta=0.0,
        amp_sigma=1e-3,
        laser_waist=float("inf"),
        runs=10,
        samples_per_run=1,
    )
    assert SimConfig.from_noise_model(noise_model) == expected_simconfig
    assert expected_simconfig.to_noise_model() == noise_model


@pytest.mark.parametrize(
    "register2D",
    [
        True,
        False,
    ],
)
def test_noisy_register(register2D: bool) -> None:
    """Testing noisy_register function."""
    if register2D:
        qdict = {
            "q0": np.array([-15.0, 0.0]),
            "q1": np.array([-5.0, 0.0]),
            "q2": np.array([5.0, 0.0]),
            "q3": np.array([15.0, 0.0]),
        }
    else:
        qdict = {
            "q0": np.array([-15.0, 0.0, 0.0]),
            "q1": np.array([-5.0, 0.0, 0.0]),
            "q2": np.array([5.0, 0.0, 0.0]),
            "q3": np.array([15.0, 0.0, 0.0]),
        }

    register_sigma_xy = 0.13
    register_sigma_z = 0.8
    # Predefined deterministic noise
    fake_normal_xy_noise = np.array(
        [
            [0.1, -0.1],
            [0.2, -0.2],
            [0.3, -0.3],
            [0.5, -0.5],
        ]
    )
    fake_normal_z_noise = np.array([0.05, 0.07, 0.09, 0.11])
    with patch("numpy.random.normal") as mock_normal:
        # moke the noise generation
        mock_normal.side_effect = [fake_normal_xy_noise, fake_normal_z_noise]

        result = noisy_register(qdict, register_sigma_xy, register_sigma_z)

    expected_positions = {
        "q0": np.array([-15.0 + 0.1, 0.0 - 0.1, 0.05]),
        "q1": np.array([-5.0 + 0.2, 0.0 - 0.2, 0.07]),
        "q2": np.array([5.0 + 0.3, 0.0 - 0.3, 0.09]),
        "q3": np.array([15.0 + 0.5, 0.0 - 0.5, 0.11]),
    }

    for q in qdict:
        np.testing.assert_array_almost_equal(result[q], expected_positions[q])
