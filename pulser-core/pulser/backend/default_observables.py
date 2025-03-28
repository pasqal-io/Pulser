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
"""Defines the default observables."""
from __future__ import annotations

import copy
import functools
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Type

import pulser.math as pm
from pulser.backend.observable import Observable
from pulser.backend.operator import Operator, OperatorType
from pulser.backend.state import Eigenstate, State, StateType
from pulser.exceptions.serialization import AbstractReprError

if TYPE_CHECKING:
    from pulser.backend.config import EmulationConfig


class StateResult(Observable):
    """Stores the quantum state at the evaluation times.

    Args:
        evaluation_times: The relative times at which to store the state.
            If left as `None`, uses the ``default_evaluation_times`` of the
            backend's ``EmulationConfig``.
        tag_suffix: An optional suffix to append to the tag. Needed if
            multiple instances of the same observable are given to the
            same EmulationConfig.
    """

    @property
    def _base_tag(self) -> str:
        return "state"

    def _to_abstract_repr(self) -> dict[str, Any]:
        raise AbstractReprError(
            """`StateResult` observable is not supported in any remote backend.
            If you are interested in the full quantum state at arbitrary times
            during the emulation, please, consider using the local version of
            the same backend."""
        )

    def apply(self, *, state: StateType, **kwargs: Any) -> StateType:
        """Calculates the observable to store in the Results."""
        return copy.deepcopy(state)


class BitStrings(Observable):
    """Stores bitstrings sampled from the state at the evaluation times.

    Error rates are taken from the NoiseModel passed to the backend via
    the EmulationConfig.
    The bitstrings are stored as a Counter[str].

    Args:
        evaluation_times: The relative times at which to sample bitstrings.
            If left as `None`, uses the ``default_evaluation_times`` of the
            backend's ``EmulationConfig``.
        num_shots: How many bitstrings to sample each time this observable
            is computed.
        one_state: The eigenstate that measures to 1. Can be left undefined
            if the state's eigenstates form a known eigenbasis with a
            defined "one state".
        tag_suffix: An optional suffix to append to the tag. Needed if
            multiple instances of the same observable are given to the
            same EmulationConfig.
    """

    def __init__(
        self,
        *,
        evaluation_times: Sequence[float] | None = None,
        num_shots: int = 1000,
        one_state: Eigenstate | None = None,
        tag_suffix: str | None = None,
    ):
        """Initializes the observable."""
        super().__init__(
            evaluation_times=evaluation_times, tag_suffix=tag_suffix
        )
        if num_shots < 1:
            raise ValueError(
                "'num_shots' must be greater than or equal to 1, "
                f"not {num_shots}."
            )
        self.num_shots = int(num_shots)
        self.one_state = one_state

    @property
    def _base_tag(self) -> str:
        return "bitstrings"

    def _to_abstract_repr(self) -> dict[str, Any]:
        repr = super()._to_abstract_repr()
        repr["num_shots"] = self.num_shots
        repr["one_state"] = self.one_state
        return repr

    def apply(
        self,
        *,
        config: EmulationConfig,
        state: State,
        **kwargs: Any,
    ) -> Counter[str]:
        """Calculates the observable to store in the Results."""
        return state.sample(
            num_shots=self.num_shots,
            one_state=self.one_state,
            p_false_pos=config.noise_model.p_false_pos,
            p_false_neg=config.noise_model.p_false_neg,
        )


class Fidelity(Observable):
    """Stores the fidelity with a pure state at the evaluation times.

    The fidelity uses the overlap between the given state and the state of
    the system at each evaluation time. For pure states, this corresponds
    to ``|<ψ|φ(t)>|^2`` for the given state ``|ψ>`` and the state ``|φ(t)>``
    obtained by time evolution.

    Args:
        state: The state ``|ψ>``. Note that this must be of an appropriate type
            for the backend.
        evaluation_times: The relative times at which to compute the fidelity.
            If left as `None`, uses the ``default_evaluation_times`` of the
            backend's ``EmulationConfig``.
        tag_suffix: An optional suffix to append to the tag. Needed if
            multiple instances of the same observable are given to the
            same EmulationConfig.
    """

    def __init__(
        self,
        state: State,
        *,
        evaluation_times: Sequence[float] | None = None,
        tag_suffix: str | None = None,
    ):
        """Initializes the observable."""
        super().__init__(
            evaluation_times=evaluation_times, tag_suffix=tag_suffix
        )
        if not isinstance(state, State):
            raise TypeError(
                f"'state' must be a State instance; got {type(state)} instead."
            )
        self.state = state

    @property
    def _base_tag(self) -> str:
        return "fidelity"

    def _to_abstract_repr(self) -> dict[str, Any]:
        repr = super()._to_abstract_repr()
        repr["state"] = self.state
        return repr

    def apply(self, *, state: State, **kwargs: Any) -> Any:
        """Calculates the observable to store in the Results."""
        return self.state.overlap(state)


class Expectation(Observable):
    """Stores the expectation of the given operator on the current state.

    Args:
        evaluation_times: The relative times at which to compute the
            expectation value. If left as `None`, uses the
            ``default_evaluation_times`` of the backend's ``EmulationConfig``.
        operator: The operator to measure. Must be of the appropriate type
            for the backend.
        tag_suffix: An optional suffix to append to the tag. Needed if
            multiple instances of the same observable are given to the
            same EmulationConfig.
    """

    def __init__(
        self,
        operator: Operator,
        *,
        evaluation_times: Sequence[float] | None = None,
        tag_suffix: str | None = None,
    ):
        """Initializes the observable."""
        super().__init__(
            evaluation_times=evaluation_times, tag_suffix=tag_suffix
        )
        if not isinstance(operator, Operator):
            raise TypeError(
                "'operator' must be an Operator instance;"
                f" got {type(operator)} instead."
            )
        self.operator = operator

    @property
    def _base_tag(self) -> str:
        return "expectation"

    def _to_abstract_repr(self) -> dict[str, Any]:
        repr = super()._to_abstract_repr()
        repr["operator"] = self.operator
        return repr

    def apply(self, *, state: State, **kwargs: Any) -> Any:
        """Calculates the observable to store in the Results."""
        return self.operator.expect(state)


class CorrelationMatrix(Observable):
    """Stores the correlation matrix for the current state.

    The correlation matrix is calculated as
    ``[[<φ(t)|n_i n_j|φ(t)> for j in qubits] for i in qubits]``
    where ``n_k = |one_state><one_state|``.

    Args:
        evaluation_times: The relative times at which to compute the
            correlation matrix. If left as `None`, uses the
            ``default_evaluation_times`` of the backend's ``EmulationConfig``.
        one_state: The eigenstate to measure the population of in the
            correlation matrix. Can be left undefined if the state's
            eigenstates form a known eigenbasis with a defined "one state".
        tag_suffix: An optional suffix to append to the tag. Needed if
            multiple instances of the same observable are given to the
            same EmulationConfig.
    """

    def __init__(
        self,
        *,
        evaluation_times: Sequence[float] | None = None,
        one_state: Eigenstate | None = None,
        tag_suffix: str | None = None,
    ):
        """Initializes the observable."""
        super().__init__(
            evaluation_times=evaluation_times, tag_suffix=tag_suffix
        )
        self.one_state = one_state

    @property
    def _base_tag(self) -> str:
        return "correlation_matrix"

    def _to_abstract_repr(self) -> dict[str, Any]:
        repr = super()._to_abstract_repr()
        repr["one_state"] = self.one_state
        return repr

    @staticmethod
    @functools.cache
    def _get_number_operator(
        qudit_ids: frozenset[int],
        n_qudits: int,
        eigenstates: Sequence[Eigenstate],
        one_state: Eigenstate,
        op_type: Type[OperatorType],
    ) -> OperatorType:
        n_op = {one_state * 2: 1.0}
        return op_type.from_operator_repr(
            eigenstates=eigenstates,
            n_qudits=n_qudits,
            operations=[(1.0, [(n_op, qudit_ids)])],
        )

    def apply(
        self, *, state: State, hamiltonian: Operator, **kwargs: Any
    ) -> list[list]:
        """Calculates the observable to store in the Results."""

        @functools.cache
        def calc_expectation(qudit_ids: frozenset[int]) -> Any:
            return self._get_number_operator(
                qudit_ids,
                state.n_qudits,
                state.eigenstates,
                self.one_state or state.infer_one_state(),
                type(hamiltonian),
            ).expect(state)

        return [
            [
                calc_expectation(frozenset((i, j)))
                for j in range(state.n_qudits)
            ]
            for i in range(state.n_qudits)
        ]


class Occupation(Observable):
    """Stores the occupation number of an eigenstate on each qudit.

    For every qudit i, calculates ``<φ(t)|n_i|φ(t)>``, where
    ``n_i = |one_state><one_state|``.

    Args:
        evaluation_times: The relative times at which to compute the
            correlation matrix. If left as ``None``, uses the
            ``default_evaluation_times`` of the backend's ``EmulationConfig``.
        one_state: The eigenstate to measure the population of. Can be left
            undefined if the state's eigenstates form a known eigenbasis with
            a defined "one state".
        tag_suffix: An optional suffix to append to the tag. Needed if
            multiple instances of the same observable are given to the
            same EmulationConfig.
    """

    def __init__(
        self,
        *,
        evaluation_times: Sequence[float] | None = None,
        one_state: Eigenstate | None = None,
        tag_suffix: str | None = None,
    ):
        """Initializes the observable."""
        super().__init__(
            evaluation_times=evaluation_times, tag_suffix=tag_suffix
        )
        self.one_state = one_state

    @property
    def _base_tag(self) -> str:
        return "occupation"

    def _to_abstract_repr(self) -> dict[str, Any]:
        repr = super()._to_abstract_repr()
        repr["one_state"] = self.one_state
        return repr

    def apply(
        self, *, state: State, hamiltonian: Operator, **kwargs: Any
    ) -> list:
        """Calculates the observable to store in the Results."""
        return [
            CorrelationMatrix._get_number_operator(
                frozenset((i,)),
                state.n_qudits,
                state.eigenstates,
                self.one_state or state.infer_one_state(),
                type(hamiltonian),
            ).expect(state)
            for i in range(state.n_qudits)
        ]


class Energy(Observable):
    """Stores the energy of the system at the evaluation times.

    The energy is calculated as the expectation value of the Hamiltonian,
    i.e. ``<φ(t)|H(t)|φ(t)>``.

    Args:
        evaluation_times: The relative times at which to compute the energy.
            If left as `None`, uses the ``default_evaluation_times`` of the
            backend's ``EmulationConfig``.
        tag_suffix: An optional suffix to append to the tag. Needed if
            multiple instances of the same observable are given to the
            same EmulationConfig.
    """

    @property
    def _base_tag(self) -> str:
        return "energy"

    def apply(
        self, *, state: State, hamiltonian: Operator, **kwargs: Any
    ) -> Any:
        """Calculates the observable to store in the Results."""
        return hamiltonian.expect(state)


class EnergyVariance(Observable):
    r"""Stores the variance of the Hamiltonian at the evaluation times.

    The variance of the Hamiltonian at time ``t`` is calculated by
    ``<φ(t)|H(t)^2|φ(t)> - <φ(t)|H(t)|φ(t)>^2``


    Args:
        evaluation_times: The relative times at which to compute the variance.
            If left as `None`, uses the ``default_evaluation_times`` of the
            backend's ``EmulationConfig``.
        tag_suffix: An optional suffix to append to the tag. Needed if
            multiple instances of the same observable are given to the
            same EmulationConfig.
    """

    @property
    def _base_tag(self) -> str:
        return "energy_variance"

    def apply(
        self, *, state: State, hamiltonian: Operator, **kwargs: Any
    ) -> Any:
        """Calculates the observable to store in the Results."""
        # This works for state vectors and density matrices and avoids
        # squaring the hamiltonian
        h_state = hamiltonian.apply_to(state)
        result: pm.AbstractArray = pm.sqrt(
            h_state.overlap(h_state).real
        ) - state.overlap(h_state)
        if not result.requires_grad:
            return float(result)
        # If the result requires_grad, return the AbstractArray
        return result  # pragma: no cover


class EnergySecondMoment(Observable):
    """Stores the expectation value of ``H(t)^2`` at the evaluation times.

    Useful for computing the variance when averaging over many executions of
    the program.

    Args:
        evaluation_times: The relative times at which to compute the variance.
            If left as `None`, uses the ``default_evaluation_times`` of the
            backend's ``EmulationConfig``.
        tag_suffix: An optional suffix to append to the tag. Needed if
            multiple instances of the same observable are given to the
            same EmulationConfig.
    """

    @property
    def _base_tag(self) -> str:
        return "energy_second_moment"

    def apply(
        self, *, state: State, hamiltonian: Operator, **kwargs: Any
    ) -> Any:
        """Calculates the observable to store in the Results."""
        # This works for state vectors and density matrices and avoids
        # squaring the hamiltonian
        h_state = hamiltonian.apply_to(state)
        result = pm.sqrt(h_state.overlap(h_state).real)
        if not result.requires_grad:
            return float(result)
        return result  # pragma: no cover
