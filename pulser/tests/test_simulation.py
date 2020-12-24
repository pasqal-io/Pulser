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

from pulser import Sequence, Pulse, Register
from pulser.devices import Chadoq2, MockDevice
from pulser.devices._pasqal_device import PasqalDevice
from pulser.sequence import TimeSlot
from pulser.waveforms import BlackmanWaveform
from pulser.simulation import Simulation

reg = Register.triangular_lattice(4, 7, spacing=5, prefix='q')
device = Chadoq2


def test_init():
    fake_sequence = {'pulse1': 'fake', 'pulse2': "fake"}
    with pytest.raises(TypeError, match='sequence has to be a valid'):
        Simulation(fake_sequence)
