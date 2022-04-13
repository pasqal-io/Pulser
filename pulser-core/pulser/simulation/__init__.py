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
"""Classes for classical emulation of a Sequence."""

import warnings

main_msg = (
    "All features previously located in the 'pulser.simulation' module have "
    "been moved to the 'pulser_simulation' package. "
)

try:
    import pulser_simulation
    from pulser_simulation.simconfig import SimConfig
    from pulser_simulation.simulation import Simulation

    warnings.warn(
        main_msg + "It is recommended that all imports of/from "
        "'pulser.simulation' are changed to 'pulser_simulation'.",
        stacklevel=2,
    )
except ImportError:  # pragma: no cover
    raise ImportError(
        main_msg + "Please install the 'pulser_simulation' package and import"
        " all simulation-related objects directly from it. "
    )
