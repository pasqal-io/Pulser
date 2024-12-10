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
"""Defines the base class for storing backend results."""
from __future__ import annotations

import collections.abc
import typing
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

from pulser.register import QubitId


@dataclass
class Results:
    """A collection of results.

    Args:
        atoms_order: The order of the atoms/qudits in the results.
        total_duration: The total duration of the sequence, in ns.
    """

    atom_order: tuple[QubitId, ...]
    total_duration: int
    _results: dict[str, dict[float, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self._results = {}

    def _store(self, *, observable_name: str, time: float, value: Any) -> None:
        """Store the result of an observable at a specific time.

        Args:
            observable_name: The name of the observable under which to store
                the result.
            time: The relative time at which the observable was taken.
            value: The value of the observable.
        """
        self._results.setdefault(observable_name, {})

        if time in self._results[observable_name]:
            raise ValueError(
                f"A value is already stored for observable '{observable_name}'"
                f" at time {time}."
            )

        self._results[observable_name][time] = value

    def __getattr__(self, name: str) -> Any:
        if name in self._results:
            return dict(self._results[name])
        raise AttributeError(f"{name!r} is not in the results.")

    def get_result_names(self) -> list[str]:
        """Get a list of results names present in this object."""
        return list(self._results.keys())

    def get_result_times(self, name: str) -> list[float]:
        """Get a list of times for which the given result has been stored.

        Args:
            name: Name of the result to get times of.

        Returns:
            List of relative times.
        """
        return list(getattr(self, name).keys())

    def get_result(self, name: str, time: float) -> Any:
        """Get the given result at the given time.

        Args:
            name: Name of the result to get.
            time: Relative time at which to get the result.

        Returns:
            The result.
        """
        try:
            return getattr(self, name)[time]
        except KeyError:
            raise ValueError(f"{name!r} is not available at time {time}.")


ResultsType = TypeVar("ResultsType", bound=Results)


class ResultsSequence(typing.Sequence[ResultsType]):
    """An immutable sequence of results."""

    _results_seq: tuple[ResultsType, ...]

    @overload
    def __getitem__(self, key: int) -> ResultsType:
        pass

    @overload
    def __getitem__(self, key: slice) -> tuple[ResultsType, ...]:
        pass

    def __getitem__(
        self, key: int | slice
    ) -> ResultsType | tuple[ResultsType, ...]:
        return self._results_seq[key]

    def __len__(self) -> int:
        return len(self._results_seq)

    def __iter__(self) -> collections.abc.Iterator[ResultsType]:
        for res in self._results_seq:
            yield res
