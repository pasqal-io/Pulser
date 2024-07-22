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
"""A class to define the Pulser Hamiltonian."""
from __future__ import annotations

from pulser.operators.operator import TimeOperator
from pulser import Sequence
from pulser.register.base_register import BaseRegister
from pulser.devices._device_datacls import BaseDevice
from pulser.sampler.samples import SequenceSamples


class Hamiltonian(TimeOperator):

    @classmethod
    def from_samples(
        cls,
        sampled_seq: SequenceSamples,
        register: BaseRegister,
        device: BaseDevice,
        sampling_rate: float = 1.0,
    ) -> Hamiltonian:
        pass

    @classmethod
    def from_sequence(cls, sequence: Sequence) -> Hamiltonian:
        pass
