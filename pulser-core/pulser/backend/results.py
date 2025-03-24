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
from typing import Any, TypeVar, overload

from pulser.backend.observable import Observable


@dataclass
class Results:
    """A collection of results.

    Args:
        atom_order: The order of the atoms/qudits in the results.
        total_duration: The total duration of the sequence, in ns.
    """

    atom_order: tuple[str, ...]
    total_duration: int
    _results: dict[uuid.UUID, list[Any]] = field(init=False)
    _times: dict[uuid.UUID, list[float]] = field(init=False)
    _tagmap: dict[str, uuid.UUID] = field(init=False)

    def __post_init__(self) -> None:
        self._results = {}
        self._times = {}
        self._tagmap = {}

    def _store_raw(
        self, *, uuid: uuid.UUID, tag: str, time: float, value: Any
    ) -> None:
        _times = self._times.setdefault(uuid, [])
        if time in _times:
            raise RuntimeError(
                f"A value is already stored for observable '{tag}'"
                f" at time {time}."
            )
        self._tagmap[tag] = uuid
        assert (
            _times == [] or _times[-1] < time
        ), "Evaluation times are not sorted."
        _times.append(time)
        self._results.setdefault(uuid, []).append(value)
        assert len(_times) == len(self._results[uuid])

    def _store(
        self, *, observable: Observable, time: float, value: Any
    ) -> None:
        """Store the result of an observable at a specific time.

        Args:
            observable: The observable computing the result.
            time: The relative time at which the observable was taken.
            value: The value of the observable.
        """
        self._store_raw(observable.uuid, observable.tag, time, value)

    def __getattr__(self, name: str) -> list[Any]:
        if name in self._tagmap:
            return list(self._results[self._tagmap[name]])
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
        return list(self._times[self._find_uuid(observable)])

    def get_result(self, observable: Observable | str, time: float) -> Any:
        """Get the a specific result at a given time.

        Args:
            observable: The observable instance used to calculate the result
                or its tag.
            time: Relative time at which to get the result.

        Returns:
            The result.
        """
        obs_uuid = self._find_uuid(observable)
        try:
            ind = self._times[obs_uuid].index(time)
            return self._results[obs_uuid][ind]
        except (KeyError, ValueError):
            raise ValueError(
                f"{observable!r} is not available at time {time}."
            )

    def get_tagged_results(self) -> dict[str, list[Any]]:
        """Gets the results for every tag.

        Returns:
            A mapping between a tag and the results associated to it,
            at every evaluation time.
        """
        return {
            tag: list(self._results[uuid_])
            for tag, uuid_ in self._tagmap.items()
        }

    def _find_uuid(self, observable: Observable | str) -> uuid.UUID:
        if isinstance(observable, Observable):
            if observable.uuid not in self._results:
                raise ValueError(
                    f"'{observable!r}' has not been stored in the results"
                )
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
