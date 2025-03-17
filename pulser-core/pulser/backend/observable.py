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

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from pulser.backend.operator import Operator
from pulser.backend.state import State

if TYPE_CHECKING:
    from pulser.backend.config import EmulationConfig
    from pulser.backend.results import Results


class Callback(ABC):
    """A general Callback that is called during the emulation."""

    def __init__(self) -> None:
        """Initializes a Callback."""
        self._uuid: uuid.UUID = uuid.uuid4()

    @property
    def uuid(self) -> uuid.UUID:
        """A universal unique identifier for this instance."""
        return self._uuid

    @abstractmethod
    def __call__(
        self,
        config: EmulationConfig,
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
            If left as `None`, uses the ``default_evaluation_times`` of the
            backend's ``EmulationConfig``.
        tag_suffix: An optional suffix to append to the tag. Needed if
            multiple instances of the same observable are given to the
            same EmulationConfig.
    """

    def __init__(
        self,
        *,
        evaluation_times: Sequence[float] | None = None,
        tag_suffix: str | None = None,
    ):
        """Initializes the observable."""
        super().__init__()
        if evaluation_times is not None:
            self._validate_eval_times(evaluation_times)
        self.evaluation_times = evaluation_times
        self._tag_suffix = tag_suffix

    @property
    @abstractmethod
    def _base_tag(self) -> str:
        pass

    def _to_abstract_repr(self) -> dict[str, Any]:
        return {
            "observable": self._base_tag,
            "evaluation_times": self.evaluation_times,
            "tag_suffix": self._tag_suffix,
        }

    @property
    def tag(self) -> str:
        """Label for the observable, used to index the Results object.

        Within a Results instance, all computed observables must have different
        tags.

        Returns:
            The tag of the observable.
        """
        if self._tag_suffix is None:
            return self._base_tag
        return f"{self._base_tag}_{self._tag_suffix}"

    def __call__(
        self,
        config: EmulationConfig,
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
            time_tol: Tolerance below which two time values are considered
                equal.
        """
        time_tol = (
            (0.5 / result.total_duration) if result.total_duration else 1e-6
        )
        if (
            self.evaluation_times is not None
            and config.is_time_in_evaluation_times(
                t, self.evaluation_times, tol=time_tol
            )
        ) or config.is_evaluation_time(t, tol=time_tol):
            value_to_store = self.apply(
                config=config, state=state, hamiltonian=hamiltonian
            )
            result._store(observable=self, time=t, value=value_to_store)

    @abstractmethod
    def apply(
        self,
        *,
        config: EmulationConfig,
        state: State,
        hamiltonian: Operator,
    ) -> Any:
        """Calculates the observable to store in the Results.

        Args:
            config: The config object passed to the backend.
            state: The current state.
            hamiltonian: The Hamiltonian at this time.

        Returns:
            The result to put in Results.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.tag}:{self.uuid}"

    @staticmethod
    def _validate_eval_times(
        evaluation_times: ArrayLike | Sequence[float],
    ) -> np.ndarray:
        eval_times_arr = np.array(evaluation_times, dtype=float)
        if np.any((eval_times_arr < 0.0) | (eval_times_arr > 1.0)):
            raise ValueError(
                "All evaluation times must be between 0. and 1. "
                f"Instead, got {evaluation_times!r}."
            )
        unique_eval_times = np.unique(eval_times_arr)
        if unique_eval_times.size < eval_times_arr.size:
            raise ValueError(
                "Evaluation times must be unique but "
                f"{evaluation_times!r} has repeated values."
            )
        if not np.all(eval_times_arr[:-1] < eval_times_arr[1:]):
            raise ValueError(
                "Evaluation times must be in ascending order."
                f"Instead, got {evaluation_times!r}."
            )
        return eval_times_arr
