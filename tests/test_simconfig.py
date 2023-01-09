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

from pulser_simulation import SimConfig


@pytest.fixture
def matrices():
    pauli = {}
    pauli["I"] = qeye(2)
    pauli["X"] = sigmax()
    pauli["Zh"] = 0.5 * sigmaz()
    pauli["ket"] = Qobj([[1.0], [2.0]])
    pauli["I3"] = qeye(3)
    return pauli


def test_init(matrices):
    config = SimConfig(
        noise=(
            "SPAM",
            "doppler",
            "dephasing",
            "amplitude",
            "depolarizing",
            "gen_noise",
        ),
        gen_noise_opers=[matrices["I"]],
        gen_noise_probs=[1.0],
        temperature=1000.0,
        runs=100,
    )
    str_config = config.__str__(True)
    assert "SPAM, doppler, dephasing, amplitude, depolarizing" in str_config
    assert (
        "1000.0µK" in str_config
        and "100" in str_config
        and "Solver Options" in str_config
    )
    assert "General noise distribution", "General noise operators"
    with pytest.raises(ValueError, match="is not a valid noise type."):
        SimConfig(noise="bad_noise")
    with pytest.raises(ValueError, match="Temperature field"):
        SimConfig(temperature=-1.0)
    with pytest.raises(ValueError, match="SPAM parameter"):
        SimConfig(eta=-1.0)
    with pytest.raises(
        ValueError, match="The standard deviation in amplitude"
    ):
        SimConfig(amp_sigma=-0.001)
    with pytest.raises(ValueError, match="The operators list length"):
        SimConfig(gen_noise_probs=[1.0])
    with pytest.raises(ValueError, match="Fill the general noise parameters."):
        SimConfig(noise=("gen_noise"))
    with pytest.raises(TypeError, match="gen_noise_probs is a list of floats"):
        SimConfig(
            noise=("gen_noise"), gen_noise_probs=[""], gen_noise_opers=[""]
        )
    with pytest.raises(ValueError, match="is not a probability distribution."):
        SimConfig(
            noise=("gen_noise"),
            gen_noise_opers=[matrices["I"], matrices["X"]],
            gen_noise_probs=[-1.0, 0.5],
        )
    with pytest.raises(ValueError, match="is not a probability distribution."):
        SimConfig(
            noise=("gen_noise"),
            gen_noise_opers=[matrices["I"], matrices["X"]],
            gen_noise_probs=[0.5, 2.0],
        )
    with pytest.raises(ValueError, match="is not a probability distribution."):
        SimConfig(
            noise=("gen_noise"),
            gen_noise_opers=[matrices["I"], matrices["X"]],
            gen_noise_probs=[0.3, 0.2],
        )
    with pytest.raises(TypeError, match="is not a Qobj."):
        SimConfig(
            noise=("gen_noise"), gen_noise_opers=[2.0], gen_noise_probs=[1.0]
        )
    with pytest.raises(TypeError, match="to be of type oper."):
        SimConfig(
            noise=("gen_noise"),
            gen_noise_opers=[matrices["ket"]],
            gen_noise_probs=[1.0],
        )
    with pytest.raises(NotImplementedError, match="Operator's shape"):
        SimConfig(
            noise=("gen_noise"),
            gen_noise_opers=[matrices["I3"]],
            gen_noise_probs=[1.0],
        )
    with pytest.raises(
        NotImplementedError, match="You must put the identity matrix"
    ):
        SimConfig(
            noise=("gen_noise"),
            gen_noise_opers=[matrices["X"], matrices["I"]],
            gen_noise_probs=[0.5, 0.5],
        )
    with pytest.raises(ValueError, match="The completeness relation is not"):
        SimConfig(
            noise=("gen_noise"),
            gen_noise_opers=[matrices["I"], matrices["Zh"]],
            gen_noise_probs=[0.5, 0.5],
        )
