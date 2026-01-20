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
"""Defines aggregation functions specific to pulser-simulation."""

from pulser_simulation.qutip_state import QutipState


def density_matrix_aggregator(values: list[QutipState]) -> QutipState:
    """Average states into a mixed state.

    Argument:
        values: The results to average.

    Returns:
        The mixed state containing each input state with probability 1/n.
    """
    acc = values[0]._state
    if not acc.isoper:
        acc = acc * acc.dag()
    for state in values[1:]:
        if not state._state.isoper:
            q_state = state._state * state._state.dag()
        else:
            q_state = state._state
        acc += q_state
    return QutipState(acc / len(values), eigenstates=values[0].eigenstates)
