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
    config = SimConfig(noise=('SPAM', 'doppler'), temperature=1000., runs=100)
    str_config = str(config)
    assert "SPAM, doppler" in str_config and "0.001K" in str_config and \
        "100" in str_config
    with pytest.raises(ValueError, match="is not a valid noise type."):
        SimConfig(noise='bad_noise')
    with pytest.raises(ValueError, match="Temperature field"):
        SimConfig(temperature=-1.)
    with pytest.raises(ValueError, match="SPAM parameter"):
        SimConfig(eta=-1.)
