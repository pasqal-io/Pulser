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
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, overload

from pulser.register import QubitId

if TYPE_CHECKING:
    from pulser.backend.observable import Observable


@dataclass
class Results:
    """A collection of results.

    Args:
        atoms_order: The order of the atoms/qudits in the results.
        total_duration: The total duration of the sequence, in ns.
    """

    atom_order: tuple[QubitId, ...]
    total_duration: int
    _results: dict[uuid.UUID, dict[float, Any]] = field(init=False)
    _tagmap: dict[str, uuid.UUID] = field(init=False)

    def __post_init__(self) -> None:
        self._results = {}
        self._tagmap = {}

    def _store(
        self, *, observable: Observable, time: float, value: Any
    ) -> None:
        """Store the result of an observable at a specific time.

        Args:
            observable: The observable computing the result.
            time: The relative time at which the observable was taken.
            value: The value of the observable.
        """
        self._results.setdefault(observable.uuid, {})

        if time in self._results[observable.uuid]:
            raise ValueError(
                f"A value is already stored for observable '{observable.tag}'"
                f" at time {time}."
            )
        self._tagmap[observable.tag] = observable.uuid
        self._results[observable.uuid][time] = value

    def __getattr__(self, name: str) -> Any:
        if name in self._tagmap:
            return dict(self._results[self._tagmap[name]])
        raise AttributeError(f"{name!r} is not in the results.")

    def get_result_tags(self) -> list[str]:
        """Get a list of results tags present in this object."""
        return list(self._tagmap.keys())

    def get_result_times(self, observable: Observable | str) -> list[float]:
        """Get a list of times for which the given result has been stored.

        Args:
            observable: The observable instance used to calculate the result
                or its tag.

        Returns:
            List of relative times.
        """
        return list(self._results[self._find_uuid(observable)].keys())

    def get_result(self, observable: Observable | str, time: float) -> Any:
        """Get the given result at the given time.

        Args:
            observable: The observable instance used to calculate the result
                or its tag.
            time: Relative time at which to get the result.

        Returns:
            The result.
        """
        try:
            return self._results[self._find_uuid(observable)][time]
        except KeyError:
            raise ValueError(
                f"{observable!r} is not available at time {time}."
            )

    def get_tagged_results(self) -> dict[str, dict[float, Any]]:
        """Gets the results for every tag.

        Returns:
            A mapping between a tag and the results associated to it,
            at every evaluation time.
        """
        return {
            tag: dict(self._results[uuid_])
            for tag, uuid_ in self._tagmap.items()
        }

    def _find_uuid(self, observable: Observable | str) -> uuid.UUID:
        if isinstance(observable, Observable):
            return observable.uuid
        try:
            return self._tagmap[observable]
        except KeyError:
            raise ValueError(
                f"{observable!r} is not an Observable instance "
                "nor a known observable tag in the results."
            )


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
