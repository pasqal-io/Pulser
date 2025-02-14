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
"""Classes for backend execution."""

import pulser.noise_model as noise_model  # For backwards compat
from pulser.backend.abc import Backend, EmulatorBackend
from pulser.backend.config import EmulatorConfig, EmulationConfig
from pulser.noise_model import NoiseModel as NoiseModel  # For backwards compat
from pulser.backend.qpu import QPUBackend
from pulser.backend.results import Results
from pulser.backend.state import State
from pulser.backend.operator import Operator
from pulser.backend.observable import Callback, Observable
from pulser.backend.default_observables import (
    BitStrings,
    CorrelationMatrix,
    Energy,
    EnergySecondMoment,
    EnergyVariance,
    Expectation,
    Fidelity,
    Occupation,
    StateResult,
)

__all__ = [
    "Backend",
    "QPUBackend",
    "EmulatorBackend",
    "EmulatorConfig",
    "EmulationConfig",
    "Results",
    "Operator",
    "State",
    "Callback",
    "Observable",
    "BitStrings",
    "CorrelationMatrix",
    "Energy",
    "EnergySecondMoment",
    "EnergyVariance",
    "Expectation",
    "Fidelity",
    "Occupation",
    "StateResult",
]
