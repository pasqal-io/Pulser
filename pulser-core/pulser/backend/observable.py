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
"""Defines the abstract base class for a callback and an observable."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pulser.backend.config import EmulatorConfig
from pulser.backend.state import State
from pulser.backend.operator import Operator
from pulser.backend.results import Results


class Callback(ABC):
    """A general Callback that is called during the emulation."""

    @abstractmethod
    def __call__(
        self,
        config: EmulatorConfig,
        t: float,
        state: State,
        hamiltonian: Operator,
        result: Results,
    ) -> None:
        """Specifies a call to the callback at a specific time.

        This is called after each time step performed by the emulator.
        By default it calls `apply()` to compute a result and put it in Results
        if t in self.evaluation_times.
        It can be overloaded to define general callbacks that don't put results
        in the Results object.

        Args:
            config: The config object passed to the backend.
            t: The relative time as a float between 0 and 1.
            state: The current state.
            hamiltonian: The Hamiltonian at this time.
            result: The Results object to store the result in.
        """
        pass


class Observable(Callback):
    """The Observable abstract base class.

    Args:
        evaluation_times: The times at which to add a result to Results.
            If left as `None`, uses the `default_evaluation_times` of the
            `EmulatorConfig` it is added to.
    """

    def __init__(self, evaluation_times: Sequence[float] | None = None):
        """Initializes the observable."""
        self.evaluation_times = evaluation_times

    def __call__(
        self,
        config: EmulatorConfig,
        t: float,
        state: State,
        hamiltonian: Operator,
        result: Results,
    ) -> None:
        """Specifies a call to the observable at a specific time.

        This is called after each time step performed by the emulator.
        By default it calls `apply()` to compute a result and put it in Results
        if t in self.evaluation_times.
        It can be overloaded to define general callbacks that don't put results
        in the Results object.

        Args:
            config: The config object passed to the backend.
            t: The relative time as a float between 0 and 1.
            state: The current state.
            hamiltonian: The Hamiltonian at this time.
            result: The Results object to store the result in.
        """
        if t in (self.evaluation_times or config.default_evaluation_times):
            value_to_store = self.apply(config, t, state, hamiltonian)
            result._store(
                observable_name=self.name(), time=t, value=value_to_store
            )

    @abstractmethod
    def name(self) -> str:
        """Name of the observable, normally used to index the Results object.

        Some Observables might have multiple instances, such as an observable
        to compute a fidelity on some given state. In that case, this method
        could make sure each instance has a unique name.

        Returns:
            The name of the observable.
        """
        pass

    @abstractmethod
    def apply(
        self,
        config: EmulatorConfig,
        t: float,
        state: State,
        hamiltonian: Operator,
    ) -> Any:
        """Calculates the observable to store in the Results.

        Args:
            config: The config object passed to the backend.
            t: The relative time as a float between 0 and 1.
            state: The current state.
            hamiltonian: The Hamiltonian at this time.

        Returns:
            The result to put in Results.
        """
        pass
