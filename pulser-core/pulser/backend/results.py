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
import json
import typing
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, cast, overload

from pulser.backend.aggregators import AGGREGATOR_MAPPING
from pulser.backend.observable import AggregationMethod, Observable
from pulser.backend.state import State
from pulser.json.abstract_repr.deserializer import deserialize_complex
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.abstract_repr.validation import validate_abstract_repr
from pulser.json.utils import stringify_qubit_ids


@dataclass(repr=False)
class Results:
    """A collection of results.

    Args:
        atom_order: The order of the atoms/qudits in the results.
        total_duration: The total duration of the sequence, in ns.
    """

    atom_order: tuple[str, ...]
    """The order of the atoms/qudits in the results."""
    total_duration: int
    """The total duration of the sequence, in ns."""
    _results: dict[uuid.UUID, list[Any]] = field(init=False, repr=False)
    _times: dict[uuid.UUID, list[float]] = field(init=False, repr=False)
    _aggregation_methods: dict[uuid.UUID, AggregationMethod] = field(
        init=False, repr=False
    )
    _tagmap: dict[str, uuid.UUID] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._results = {}
        self._times = {}
        self._tagmap = {}
        self._aggregation_methods = {}

    def _store_raw(
        self,
        *,
        uuid: uuid.UUID,
        tag: str,
        time: float,
        value: Any,
        aggregation_method: AggregationMethod,
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
        self._aggregation_methods[uuid] = aggregation_method
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
        self._store_raw(
            uuid=observable.uuid,
            tag=observable.tag,
            time=time,
            value=value,
            aggregation_method=observable.default_aggregation_method,
        )

    def __getattr__(self, name: str) -> list[Any]:
        if name in self._tagmap:
            return list(self._results[self._tagmap[name]])
        raise AttributeError(f"{name!r} is not in the results.")

    @property
    def final_bitstrings(self) -> dict[str, int]:
        """The bitstrings at the end of the sequence, if available."""
        try:
            return cast(
                typing.Dict[str, int], self.get_result("bitstrings", time=1.0)
            )
        except ValueError:
            raise RuntimeError(
                "The final bitstrings are not available. Please make sure "
                "'BitStrings()' at relative time t=1.0 is included in the "
                "observables of your emulator backend's configuration (when"
                " possible)."
            )

    @property
    def final_state(self) -> State:
        """The state at the end of the sequence, if available."""
        try:
            return cast(State, self.get_result("state", time=1.0))
        except ValueError:
            raise RuntimeError(
                "The final state is not available. Please make sure "
                "'StateResult()' at relative time t=1.0 is included in the "
                "observables of your emulator backend's configuration (when"
                " possible)."
            )

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

    def _to_abstract_repr(self) -> dict:
        d = {
            "atom_order": stringify_qubit_ids(self.atom_order),
            "total_duration": self.total_duration,
        }
        d["tagmap"] = {key: str(value) for key, value in self._tagmap.items()}
        d["results"] = {
            str(key): value for key, value in self._results.items()
        }
        d["times"] = {str(key): value for key, value in self._times.items()}
        d["aggregation_methods"] = {
            str(key): value for key, value in self._aggregation_methods.items()
        }
        return d

    @classmethod
    def _from_abstract_repr(cls, obj: dict) -> Results:
        results = cls(
            atom_order=tuple(obj["atom_order"]),
            total_duration=obj["total_duration"],
        )
        for key, value in obj["tagmap"].items():
            results._tagmap[key] = uuid.UUID(value)
        for key, value in obj["results"].items():
            results._results[uuid.UUID(key)] = deserialize_complex(value)
        for key, value in obj["times"].items():
            results._times[uuid.UUID(key)] = value
        for key, value in obj.get("aggregation_methods", {}).items():
            results._aggregation_methods[uuid.UUID(key)] = AggregationMethod(
                value
            )
        return results

    def to_abstract_repr(self, skip_validation: bool = False) -> str:
        """Serializes this object into a json string.

        Numpy arrays and torch Tensors are converted into lists,
        and their original class is lost forever.

        Args:
            skip_validation: Whether to skip validating the json against
                the schema used for deserialization.

        Returns:
            The json string
        """
        abstr_str = json.dumps(
            self._to_abstract_repr(), cls=AbstractReprEncoder
        )
        if not skip_validation:
            validate_abstract_repr(abstr_str, "results")
        return abstr_str

    @classmethod
    def from_abstract_repr(cls, repr: str) -> Results:
        """Deserializes a Results object from json.

        Returns:
            The deserialized Results object.
        """
        validate_abstract_repr(repr, "results")
        d = json.loads(repr)
        return cls._from_abstract_repr(d)

    @classmethod
    def aggregate(
        cls,
        results_to_aggregate: typing.Sequence[Results],
        **aggregation_functions: Callable[[Any], Any],
    ) -> Results:
        """Aggregate a Sequence of Results objects into a single Results.

        This is meant to accumulate the results of several runs with
        different noise trajectories into a single averaged Results.
        By default, results are averaged, with the exception of BitStrings,
        where the counters are joined.
        StateResult and EnergyVariance are not supported by default.

        Args:
            results_to_aggregate: The list of Results to aggregate

        Keyword Args:
            observable_tag: Overrides the default aggregator.
                The argument name should be the tag of the Observable.
                The value is a Callable taking a list of the type to aggregate.
                Note that this does not override the default aggregation
                behaviour of the aggregated results.

        Returns:
            The averaged Results object
        """
        if len(results_to_aggregate) == 0:
            raise ValueError("No results to aggregate.")
        result_0 = results_to_aggregate[0]
        if len(results_to_aggregate) == 1:
            return result_0

        all_tags = set().union(
            *[set(x.get_result_tags()) for x in results_to_aggregate]
        )
        common_tags = all_tags.intersection(
            *[set(x.get_result_tags()) for x in results_to_aggregate]
        )

        for results in results_to_aggregate:
            if results._results and (not results._aggregation_methods):
                raise NotImplementedError(
                    (
                        "You're trying to aggregate results from pulser<1.6,"
                        "aggregation is not supported in this case."
                    )
                )
            for tag, uid in results._tagmap.items():
                if tag not in common_tags and not (
                    results._aggregation_methods[uid].value
                    in (AggregationMethod.SKIP, AggregationMethod.SKIP_WARN)
                ):
                    raise ValueError(
                        "You're trying to aggregate incompatible results: "
                        f"result `{tag}` is not present in all results, "
                        "but it's not marked to be skipped."
                    )
        if not all(
            {
                tag: results._aggregation_methods[results._find_uuid(tag)]
                for tag in common_tags
            }
            == {
                tag: result_0._aggregation_methods[result_0._find_uuid(tag)]
                for tag in common_tags
            }
            for results in results_to_aggregate
        ):
            raise ValueError(
                "You're trying to aggregate incompatible results: "
                "they do not all contain the same aggregation functions."
            )
        if not all(
            results.atom_order == result_0.atom_order
            for results in results_to_aggregate
        ):
            raise ValueError(
                "You're trying to aggregate incompatible results: "
                "they do not all have the same atom order."
            )
        if not all(
            results.total_duration == result_0.total_duration
            for results in results_to_aggregate
        ):
            raise ValueError(
                "You're trying to aggregate incompatible results: "
                "they do not all have the same sequence duration."
            )
        aggregated = Results(
            atom_order=result_0.atom_order,
            total_duration=result_0.total_duration,
        )
        for tag in common_tags:
            default_aggregation_method = result_0._aggregation_methods[
                result_0._tagmap[tag]
            ]
            aggregation_method = aggregation_functions.get(
                tag, default_aggregation_method
            )
            if (
                aggregation_method is AggregationMethod.SKIP
                or aggregation_method is AggregationMethod.SKIP_WARN
            ):
                if aggregation_method is AggregationMethod.SKIP_WARN:
                    with warnings.catch_warnings():
                        warnings.simplefilter("once")
                        warnings.warn(f"Skipping aggregation of `{tag}`.")
                continue
            aggregation_function: Any = (
                AGGREGATOR_MAPPING[aggregation_method]
                if isinstance(aggregation_method, AggregationMethod)
                else aggregation_method
            )
            evaluation_times = results_to_aggregate[0].get_result_times(tag)
            if not all(
                results.get_result_times(tag) == evaluation_times
                for results in results_to_aggregate
            ):
                raise ValueError(
                    "The Results come from "
                    "incompatible simulations: "
                    f"the times for `{tag}` are not all the same."
                )
            uid = uuid.uuid4()

            for t in result_0.get_result_times(tag):
                v = aggregation_function(
                    [
                        result.get_result(tag, t)
                        for result in results_to_aggregate
                    ]
                )

                aggregated._store_raw(
                    uuid=uid,
                    tag=tag,
                    time=t,
                    value=v,
                    aggregation_method=default_aggregation_method,
                )

        return aggregated

    def __str__(self) -> str:
        evaluation_times = {
            tag: self._times[_uuid] for tag, _uuid in self._tagmap.items()
        }

        cls_name = self.__class__.__name__
        lines = [
            cls_name,
            "-" * len(cls_name),  # Separator
            f"Stored results: {self.get_result_tags()}",
            f"Evaluation times per result: {evaluation_times}",
            f"Atom order in states and bitstrings: {self.atom_order}",
            f"Total sequence duration: {self.total_duration} ns",
        ]
        return "\n".join(lines)


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
