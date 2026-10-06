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
"""Defines the abstract base class for a quantum operator."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from numbers import Integral
from typing import Any, Generic, Type, TypeVar

from pulser.backend.state import Eigenstate, State
from pulser.exceptions.serialization import AbstractReprError

ArgScalarType = TypeVar("ArgScalarType")
ReturnScalarType = TypeVar("ReturnScalarType")
StateType = TypeVar("StateType", bound=State)
OperatorType = TypeVar("OperatorType", bound="Operator")

# Generic type aliases
T = TypeVar("T")
QuditOp = Mapping[str, T]  # single qudit operator
TensorOp = Sequence[
    tuple[QuditOp[T], Collection[int]]
]  # QuditOp applied to set of qudits
FullOp = Sequence[tuple[T, TensorOp[T]]]  # weighted sum of TensorOp


class Operator(ABC, Generic[ArgScalarType, ReturnScalarType, StateType]):
    """Base class enforcing an API for quantum operator.

    Each backend will implement its own type of state and the methods below.
    """

    _eigenstates: Sequence[Eigenstate] | None
    _n_qudits: int | None
    _operations: FullOp[complex] | None

    def __init__(self) -> None:
        """Initializes an Operator."""
        self._eigenstates = None
        self._n_qudits = None
        self._operations = None

    @abstractmethod
    def apply_to(self, state: StateType, /) -> StateType:
        """Apply the operator to a state.

        Args:
            state: The state to apply this operator to.

        Returns:
            The resulting state.
        """
        pass

    @abstractmethod
    def expect(self, state: StateType, /) -> ReturnScalarType:
        """Compute the expectation value of self on the given state.

        Args:
            state: The state with which to compute.

        Returns:
            The expectation value.
        """
        pass

    @abstractmethod
    def __add__(self: OperatorType, other: OperatorType, /) -> OperatorType:
        """Computes the sum of two operators.

        Args:
            other: The other operator.

        Returns:
            The summed operator.
        """
        pass

    @abstractmethod
    def __rmul__(self: OperatorType, scalar: ArgScalarType) -> OperatorType:
        """Scale the operator by a scalar factor.

        Args:
            scalar: The scalar factor.

        Returns:
            The scaled operator.
        """
        pass

    @abstractmethod
    def __matmul__(self: OperatorType, other: OperatorType) -> OperatorType:
        """Compose two operators where 'self' is applied after 'other'.

        Args:
            other: The operator to compose with self.

        Returns:
            The composed operator.
        """
        pass

    @classmethod
    def from_operator_repr(
        cls: Type[OperatorType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        operations: FullOp[ArgScalarType],
    ) -> OperatorType:
        """Create an operator from the operator representation.

        Only operators constructed with this method are allowed in remote
        backend.

        The full operator representation (``FullOp``) is a weigthed sum of
        tensor operators (``TensorOp``), written as a sequence of coefficient
        and tensor operator pairs, ie

        ``FullOp = Sequence[tuple[ScalarType, TensorOp]]``

        Each ``TensorOp`` is itself a sequence of qudit operators (``QuditOp``)
        applied to mutually exclusive sets of qudits (represented by their
        indices), ie

        ``TensorOp = Sequence[tuple[QuditOp, Collection[int]]]``

        Qudits without an associated ``QuditOp`` are applied the identity
        operator.

        Finally, each ``QuditOp`` is represented as weighted sum of pre-defined
        single-qudit operators. It is given as a mapping between a string
        representation of the single-qudit operator and its respective
        coefficient, ie

        ``QuditOp = Mapping[str, ScalarType]``

        By default, it identifies strings ``"ij"`` as single-qudit operators,
        where ``i`` and ``j`` are eigenstates that denote ``|i><j|``.

        Args:
            eigenstates: The eigenstates to use.
            n_qudits: How many qudits there are in the system.
            operations: The full operator representation.

        Returns:
            The constructed operator.

        Examples:
            >>> eigenstates = ("r", "g")
            >>> n_qudits = 4
            >>> # define X,Y,Z
            >>> X = {"gr": 1.0, "rg": 1.0}
            >>> Y = {"gr": 1.0j, "rg": -1.0j}
            >>> Z = {"rr": 1.0, "gg": -1.0}
            >>> # build for example 0.5*X0Y1X2Z3 - j0.2*Z1Y3 + (-0.1+j4.0)*X2
            >>> operations = [
            >>>     (
            >>>         0.5,
            >>>         [
            >>>             (X, [0, 2]), # acts on qudit 0 and 2
            >>>             (Y, [1]),
            >>>             (Z, [3]),
            >>>         ],
            >>>     ),
            >>>     (
            >>>         -0.2j,
            >>>         [
            >>>             (Z, [1]),
            >>>             (Y, [3]),
            >>>         ],
            >>>     ),
            >>>     (
            >>>         -0.1 + 4.0j,
            >>>         [
            >>>             (X, [2]),
            >>>         ],
            >>>     ),
            >>> ]
            >>> config_class = your_pulser_backend.config_type
            >>> operator_class = config_class.operator_type
            >>> op = operator_class.from_operator_repr(
            >>>     eigenstates=eigenstates,
            >>>     n_qudits=n_qudits,
            >>>     operations=operations
            >>> )
        """
        State._validate_eigenstates(eigenstates)
        cls._validate_operations(
            eigenstates=eigenstates, n_qudits=n_qudits, operations=operations
        )
        obj, _operations = cls._from_operator_repr(
            eigenstates=eigenstates, n_qudits=n_qudits, operations=operations
        )
        obj._eigenstates = eigenstates
        obj._n_qudits = n_qudits
        obj._operations = _operations
        return obj

    @classmethod
    def from_pauli_string(
        cls: Type[OperatorType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        paulis: Mapping[int, str],
    ) -> OperatorType:
        r"""Create the operator for a Pauli string.

        A Pauli string is a product of Pauli matrices acting on different
        qudits, e.g. ``σ^z_1 σ^x_3 σ^z_10``. Qudits without an associated
        Pauli matrix are applied the identity. The resulting operator is
        built with ``from_operator_repr()``, so it is supported by every
        backend (including remote ones) and can be given to the
        ``Expectation`` observable to compute its expectation value.

        The Pauli matrices are defined following the order of the
        eigenstates. For eigenstates ``(e0, e1)``, they are

        - ``σ^x = |e0><e1| + |e1><e0|``
        - ``σ^y = -i|e0><e1| + i|e1><e0|``
        - ``σ^z = |e0><e0| - |e1><e1|``

        which matches the conventions of ``pulser-simulation`` for both the
        ``ground-rydberg`` (``("r", "g")``) and ``XY`` (``("0", "1")``)
        bases.

        The leakage state ``"x"`` (added to the eigenstates when the noise
        model includes leakage) is not one of the two states the Pauli
        matrices act on, so all Pauli matrices act as zero on it. As such,
        population in the leakage state does not contribute to the
        expectation value of a Pauli string. Bases with more than two
        eigenstates besides ``"x"`` are not supported.

        Args:
            eigenstates: The eigenstates to use. Must contain exactly two
                eigenstates, besides the optional leakage state ``"x"``.
            n_qudits: How many qudits there are in the system.
            paulis: A mapping between the index of a qudit and the Pauli
                matrix applied to it, given as one of ``"X"``, ``"Y"`` or
                ``"Z"`` (case-insensitive). The qudit indices follow the
                order of the qubit IDs in the Sequence's register (i.e.
                ``seq.register.qubit_ids``).

        Returns:
            The operator for the Pauli string.

        Example:
            >>> # σ^z_1 σ^x_3 σ^z_10 in a system of 12 qubits
            >>> operator_class = your_pulser_backend.config_type.operator_type
            >>> op = operator_class.from_pauli_string(
            >>>     eigenstates=("r", "g"),
            >>>     n_qudits=12,
            >>>     paulis={1: "Z", 3: "X", 10: "Z"},
            >>> )
            >>> # To compute its expectation value during the emulation
            >>> Expectation(op, tag_suffix="zxz")
        """
        State._validate_eigenstates(eigenstates)
        basis = [s for s in eigenstates if s != "x"]
        if len(basis) != 2:
            raise ValueError(
                "Pauli strings are only defined for qubits, i.e. with "
                "exactly two eigenstates (besides the leakage state 'x'); "
                f"got eigenstates {tuple(eigenstates)}."
            )
        if not isinstance(paulis, Mapping):
            raise TypeError(
                "'paulis' must be a mapping between qudit indices and Pauli "
                f"matrices; got {type(paulis)} instead. Got {paulis!r}."
            )
        if not paulis:
            raise ValueError("'paulis' must contain at least one entry.")
        e0, e1 = basis
        pauli_reprs: dict[str, QuditOp[complex]] = {
            "X": {e0 + e1: 1.0, e1 + e0: 1.0},
            "Y": {e0 + e1: -1.0j, e1 + e0: 1.0j},
            "Z": {e0 + e0: 1.0, e1 + e1: -1.0},
        }
        qudits_per_pauli: dict[str, set[int]] = {}
        for index, pauli in paulis.items():
            if not isinstance(index, Integral):
                raise TypeError(
                    "The qudit indices in 'paulis' must be integers; got "
                    f"{index!r} of type {type(index)}."
                )
            if not isinstance(pauli, str) or pauli.upper() not in pauli_reprs:
                raise ValueError(
                    "The Pauli matrices in 'paulis' must be one of "
                    f"{tuple(pauli_reprs)}; got {pauli!r} for qudit {index}."
                )
            qudits_per_pauli.setdefault(pauli.upper(), set()).add(int(index))
        # Invalid indices are caught when validating the operations
        return cls.from_operator_repr(
            eigenstates=eigenstates,
            n_qudits=n_qudits,
            operations=[
                (
                    1.0,
                    [
                        (pauli_reprs[pauli], qudits)
                        for pauli, qudits in qudits_per_pauli.items()
                    ],
                )
            ],
        )

    @classmethod
    @abstractmethod
    def _from_operator_repr(
        cls: Type[OperatorType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        operations: FullOp[ArgScalarType],
    ) -> tuple[OperatorType, FullOp[complex]]:
        """Implements the conversion used in `from_operator_repr()`.

        Expected to return the Operator instance alongside the 'operations' to
        use in serialization.
        """
        pass

    def _to_abstract_repr(self) -> dict[str, Any]:
        if (
            self._eigenstates is None
            or self._n_qudits is None
            or self._operations is None
        ):
            cls_name = self.__class__.__name__
            raise AbstractReprError(
                f"Failed to serialize state of type {cls_name!r} because it "
                f"was not created via '{cls_name}.from_operator_repr()'."
            )
        return {
            "eigenstates": tuple(self._eigenstates),
            "n_qudits": self._n_qudits,
            "operations": self._operations,
        }

    @staticmethod
    def _validate_operations(
        *, eigenstates: Sequence[Eigenstate], n_qudits: int, operations: FullOp
    ) -> None:
        """Check validity of operations.

        Check that operations passed as FullOp
        to `from_operator_repr` are valid.
        """
        for tensor_op_num, (coeff, tensor_op) in enumerate(operations):
            free_inds = set(range(n_qudits))
            for qudit_op, qudit_inds in tensor_op:
                if bad_inds_ := (set(qudit_inds) - free_inds):
                    raise ValueError(
                        "Got invalid indices for a system with "
                        f"{n_qudits} qudits: {bad_inds_}. For TensorOp "
                        f"#{tensor_op_num}, only indices {free_inds} "
                        "were still available."
                    )
                free_inds.difference_update(qudit_inds)

                for proj_str, coeff in qudit_op.items():
                    if len(proj_str) != 2 or any(
                        s_ not in eigenstates for s_ in proj_str
                    ):
                        raise ValueError(
                            f"Every QuditOp key must be made up"
                            f" of two eigenstates"
                            f" among {eigenstates};"
                            f" instead, got '{proj_str}'."
                        )


class OperatorRepr(Operator):
    """Define a backend-independent quantum operator representation.

    Allows the user to define a quantum operator with the dedicated class
    method `from_operator_repr`, which requires:

    - eigenstates: The basis states (e.g., ('r', 'g')).
    - n_qudits: Number of qudits in the system.
    - operations: A sequence of tuples weight, tensor operators on each qudit,
      as described in `from_operator_repr`.

    The created operator, supports de/serialization methods for remote backend
    execution.

    Examples:
        >>> eigenstates = ("r", "g")
        >>> n_qudits = 4
        >>> # define X,Y,Z
        >>> X = {"gr": 1.0, "rg": 1.0}
        >>> Y = {"gr": 1.0j, "rg": -1.0j}
        >>> Z = {"rr": 1.0, "gg": -1.0}
        >>> # build for example 0.5*X0Y1X2Z3
        >>> operations = [
        >>>     (
        >>>         0.5,
        >>>         [
        >>>             (X, [0, 2]), # acts on qudit 0 and 2
        >>>             (Y, [1]),
        >>>             (Z, [3]),
        >>>         ],
        >>>     )
        >>> ]
        >>> op = OperatorRepr.from_operator_repr(
        >>>     eigenstates=eigenstates,
        >>>     n_qudits=n_qudits,
        >>>     operations=operations
        >>> )
    """

    @classmethod
    def _from_operator_repr(
        cls: Type[OperatorType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        operations: FullOp[complex],
    ) -> tuple[OperatorType, FullOp[complex]]:
        """Implements the conversion used in `from_operator_repr()`.

        Expected to return the Operator instance alongside the 'operations' to
        use in serialization.
        """
        op = cls()
        return op, operations

    def apply_to(self, state: StateType, /) -> StateType:
        """``apply_to`` not implemented in ``OperatorRepr``."""
        raise NotImplementedError(
            "``apply_to`` not implemented in ``OperatorRepr``."
        )

    def expect(self, state: StateType, /) -> None:
        """``expect`` not implemented in ``OperatorRepr``."""
        raise NotImplementedError(
            "``expect`` not implemented in ``OperatorRepr``."
        )

    def __add__(self: OperatorType, other: OperatorType, /) -> OperatorType:
        """``__add__`` not implemented in ``OperatorRepr``."""
        raise NotImplementedError(
            "``__add__`` not implemented in ``OperatorRepr``."
        )

    def __rmul__(self: OperatorType, scalar: ArgScalarType) -> OperatorType:
        """``__rmul__`` not implemented in ``OperatorRepr``."""
        raise NotImplementedError(
            "``__rmul__`` not implemented in ``OperatorRepr``."
        )

    def __matmul__(self: OperatorType, other: OperatorType) -> OperatorType:
        """``__matmul__`` not implemented in ``OperatorRepr``."""
        raise NotImplementedError(
            "``__matmul__`` not implemented in ``OperatorRepr``."
        )
