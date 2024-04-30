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

import pytest
from qutip import Qobj, qeye, sigmax, sigmaz

from pulser.noise_model import NoiseModel
from pulser_simulation.simconfig import SimConfig, doppler_sigma


@pytest.fixture
def matrices():
    pauli = {}
    pauli["I"] = qeye(2)
    pauli["X"] = sigmax()
    pauli["Zh"] = 0.5 * sigmaz()
    pauli["ket"] = Qobj([[1.0], [2.0]])
    pauli["I3"] = qeye(3)
    return pauli


def test_init():
    config = SimConfig(
        noise=(
            "SPAM",
            "doppler",
            "dephasing",
            "amplitude",
        ),
        temperature=1000.0,
        runs=100,
    )
    assert config.temperature == 1e-3  # in K
    str_config = config.__str__(True)
    assert "SPAM, doppler, dephasing, amplitude" in str_config
    assert (
        "1000.0µK" in str_config
        and "100" in str_config
        and "Solver Options" in str_config
    )
    config = SimConfig(noise=("depolarizing", "relaxation"))
    assert config.temperature == 5e-5
    assert config.to_noise_model().temperature == 50
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
    assert config.doppler_sigma == doppler_sigma(50 * 1e-6)
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
    SimConfig(
        noise=("eff_noise"),
        eff_noise_opers=[matrices["X"], matrices["I"]],
        eff_noise_rates=[0.5, 0.5],
    )


def test_from_noise_model():
    noise_model = NoiseModel(
        noise_types=("SPAM",),
        p_false_neg=0.4,
        p_false_pos=0.1,
        state_prep_error=0.05,
    )
    assert SimConfig.from_noise_model(noise_model) == SimConfig(
        noise="SPAM", epsilon=0.1, epsilon_prime=0.4, eta=0.05
    )
