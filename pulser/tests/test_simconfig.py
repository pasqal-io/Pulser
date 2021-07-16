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

from pulser.simulation import SimConfig


def test_init():
    config = SimConfig(
        noise=("SPAM", "doppler", "dephasing", "amplitude"),
        temperature=1000.0,
        runs=100,
    )
    str_config = config.__str__(True)
    assert (
        "SPAM, doppler, dephasing, amplitude" in str_config
        and "1000.0µK" in str_config
        and "100" in str_config
        and "Solver Options" in str_config
    )
    with pytest.raises(ValueError, match="is not a valid noise type."):
        SimConfig(noise="bad_noise")
    with pytest.raises(ValueError, match="Temperature field"):
        SimConfig(temperature=-1.0)
    with pytest.raises(ValueError, match="SPAM parameter"):
        SimConfig(eta=-1.0)
