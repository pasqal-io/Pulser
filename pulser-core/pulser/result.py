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
"""Classes to store measurement results."""
from __future__ import annotations

import uuid
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import pulser.backend.results as backend_results
from pulser.backend.default_observables import BitStrings
from pulser.math.multinomial import multinomial


def __getattr__(name: str) -> Any:
    name_map = {"Results": "ResultsSequence", "ResultType": "ResultsType"}
    if name not in name_map:
        raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
    warnings.warn(
        f"The 'pulser.result.{name}' class has been renamed to "
        f"'{name_map[name]}' and moved to 'pulser.backend.results'. "
        "Importing it as '{name}' from 'pulser.results' is deprecated.",
        DeprecationWarning,
        stacklevel=3,
    )
    return getattr(backend_results, name_map[name])


__all__ = ["Result", "SampledResult"]


@dataclass
class Result(ABC, backend_results.Results):
    """Base class for storing the result of an observable at specific time."""

    meas_basis: str
    total_duration: int = field(default=0, init=False)

    @property
    def sampling_dist(self) -> dict[str, float]:
        """Sampling distribution of the measured bitstring.

        Args:
            atom_order: The order of the atoms in the bitstrings that
                represent the measured states.
            meas_basis: The measurement basis.
        """
        n = self._size
        return {
            np.binary_repr(ind, width=n): prob
            for ind, prob in enumerate(self._weights())
            if prob != 0
        }

    @property
    @abstractmethod
    def sampling_errors(self) -> dict[str, float]:
        """The sampling error associated to each bitstring's sampling rate.

        Uses the standard error of the mean as a quantifier for sampling error.
        """
        pass

    @property
    def _size(self) -> int:
        return len(self.atom_order)

    @abstractmethod
    def _weights(self) -> np.ndarray:
        """The sampling rate for every state in an ordered array."""
        pass

    def get_samples(self, n_samples: int) -> Counter[str]:
        """Takes multiple samples from the sampling distribution.

        Args:
            n_samples: Number of samples to return.

        Returns:
            Samples of bitstrings corresponding to measured quantum states.
        """
        return Counter(
            np.binary_repr(i, self._size)
            for i in multinomial(n_samples, self._weights())
        )

    def get_state(self) -> Any:
        """Gets the quantum state associated with the result.

        Can only be defined for emulation results that don't resort to
        sampling a quantum state (which is the case for certain types of
        noise).
        """
        raise NotImplementedError(
            f"`{self.__class__.__name__}.get_state()` is not implemented."
        )

    def plot_histogram(
        self,
        min_rate: float = 0.001,
        max_n_bitstrings: int | None = None,
        show: bool = True,
    ) -> None:
        """Plots the result in an histogram.

        Args:
            min_rate: The minimum sampling rate a bitstring must have to be
                displayed.
            max_n_bitstrings: An optional limit on the number of bitrstrings
                displayed.
            show: Whether or not to call `plt.show()` before returning.
        """
        # TODO: Add error bars
        probs = np.array(
            Counter(self.sampling_dist).most_common(max_n_bitstrings),
            dtype=object,
        )
        probs = probs[probs[:, 1] >= min_rate]
        plt.bar(probs[:, 0], probs[:, 1])
        plt.xticks(rotation="vertical")
        plt.ylabel("Probabilites")
        if show:
            plt.show()


@dataclass
class SampledResult(Result):
    """Represents the result of a run from a series of samples.

    Args:
        atom_order: The order of the atoms in the bitstrings that
            represent the measured states.
        meas_basis: The measurement basis.
        bitstring_counts: The number of times each bitstring was
            measured.
        evaluation_time: Relative time at which the samples were
            taken.
    """

    bitstring_counts: dict[str, int]
    evaluation_time: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_samples = sum(self.bitstring_counts.values())
        # TODO: Make sure this is not too hacky
        bitstrings_obs = BitStrings()
        # Override UUID so that two SampledResult instances with
        # the same counts are identical
        bitstrings_obs._uuid = uuid.UUID(
            "00000000-0000-0000-0000-000000000000"
        )
        self._store(
            observable=bitstrings_obs,
            time=self.evaluation_time,
            value=self.bitstring_counts,
        )

    @property
    def sampling_errors(self) -> dict[str, float]:
        """The sampling error associated to each bitstring's sampling rate.

        Uses the standard error of the mean as a quantifier for sampling error.
        """
        return {
            bitstr: np.sqrt(p * (1 - p) / self.n_samples)
            for bitstr, p in self.sampling_dist.items()
        }

    def _weights(self) -> np.ndarray:
        weights = np.zeros(2**self._size)
        for bitstr, counts in self.bitstring_counts.items():
            weights[int(bitstr, base=2)] = counts / self.n_samples
        return weights / sum(weights)
