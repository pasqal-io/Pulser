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
"""Classes for classical emulation of a Pulser Sequence."""

from pulser import EmulatorConfig, NoiseModel

from pulser_simulation._version import __version__ as __version__
from pulser_simulation.qutip_state import QutipState
from pulser_simulation.qutip_op import QutipOperator
from pulser_simulation.qutip_backend import QutipBackend, QutipBackendV2
from pulser_simulation.qutip_config import QutipConfig
from pulser_simulation.simconfig import SimConfig
from pulser_simulation.simulation import QutipEmulator

# NOTE: If any of these change, remember to MANUALLY replicate them in the
# API reference doc (ie they are not updated automatically).
__all__ = [
    "EmulatorConfig",
    "NoiseModel",
    "QutipState",
    "QutipOperator",
    "QutipConfig",
    "QutipBackend",
    "QutipBackendV2",
    "QutipEmulator",
    "SimConfig",
]
