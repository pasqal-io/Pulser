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
"""Defines a special Result subclass for simulation runs returning states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union, cast, List

import numpy as np
import qutip

from pulser.result import Result


@dataclass
class QutipResult(Result):
    """Represents the result of a run as a Qutip QObj.

    Args:
        atom_order: The order of the atoms in the bitstrings that
            represent the measured states.
        meas_basis: The measurement basis.
        state: The Qobj representing the state. Can be a statevector
            or a density matrix.
        matching_meas_basis: Whether the measurement basis is the
            same as the state's basis.
    """

    state: qutip.Qobj
    matching_meas_basis: bool

    @property
    def sampling_errors(self) -> dict[str, float]:
        """The sampling error associated to each bitstring's sampling rate.

        Uses the standard error of the mean as a quantifier for sampling error.
        """
        return {bitstr: 0.0 for bitstr in self.sampling_dist}

    @property
    def _dim(self) -> int:
        full_state_size = np.prod(self.state.shape)
        if not self.state.isket:
            full_state_size = np.sqrt(full_state_size)
        return cast(
            int, np.rint(full_state_size ** (1 / self._size)).astype(int)
        )

    @property
    def _basis_name(self) -> str:
        if self._dim > 2:
            return "all"
        if self.meas_basis == "XY":
            return "XY"
        if not self.matching_meas_basis:
            return (
                "digital"
                if self.meas_basis == "ground-rydberg"
                else "ground-rydberg"
            )
        return self.meas_basis

    def _weights(self) -> np.ndarray:
        n = self._size
        if not self.state.isket:
            probs = np.abs(self.state.diag())
        else:
            probs = (np.abs(self.state.full()) ** 2).flatten()

        if self._dim == 2:
            if self.matching_meas_basis:
                # State vector ordered with r first for 'ground_rydberg'
                # e.g. n=2: [rr, rg, gr, gg] -> [11, 10, 01, 00]
                # Invert the order ->  [00, 01, 10, 11] correspondence
                # The same applies in XY mode, which is ordered with u first
                weights = (
                    probs if self.meas_basis == "digital" else probs[::-1]
                )
            else:
                # Only 000...000 is measured
                weights = np.zeros(probs.size)
                weights[0] = 1.0

        elif self._dim == 3:
            if self.meas_basis == "ground-rydberg":
                one_state = 0  # 1 = |r>
                ex_one = slice(1, 3)
            elif self.meas_basis == "digital":
                one_state = 2  # 1 = |h>
                ex_one = slice(0, 2)
            else:
                raise RuntimeError(
                    f"Unknown measurement basis '{self.meas_basis}' "
                    "for a three-level system.'"
                )
            probs = probs.reshape([3] * n)
            weights = np.zeros(2**n)
            for dec_val in range(2**n):
                ind: list[Union[int, slice]] = []
                for v in np.binary_repr(dec_val, width=n):
                    if v == "0":
                        ind.append(ex_one)
                    else:
                        ind.append(one_state)
                # Eg: 'digital' basis : |1> = index2, |0> = index0, 1 = 0:2
                # p_11010 = sum(probs[2, 2, 0:2, 2, 0:2])
                # We sum all probabilites that correspond to measuring
                # 11010, namely hhghg, hhrhg, hhghr, hhrhr
                weights[dec_val] = np.sum(probs[tuple(ind)])
        else:
            raise NotImplementedError(
                "Cannot sample system with single-atom state vectors of "
                "dimension > 3."
            )
        # Takes care of numerical artefacts in case sum(weights) != 1
        return cast(np.ndarray, weights / sum(weights))
    
    def _get_relevant_subsystem_indices(self, to_basis: str, tol: float) -> List[int]:
        """Finds the subsystems to keep based on the target basis.

        Args:
            to_basis: The basis to which the state should be transformed.

        Returns:
            A list of indices representing the relevant subsystems.
        """
        if to_basis == "ground-rydberg":
            ex_state = "2"
        elif to_basis == "digital":
            ex_state = "0"
        else:
            raise ValueError("'to_basis' must be 'ground-rydberg' or 'digital'.")

        # Find the subsystems to keep.
        ex_inds = [
            i
            for i in range(3**self._size)
            if ex_state in np.base_repr(i, base=3).zfill(self._size)
        ]

        ex_probs = np.abs(self.state.extract_states(ex_inds).full()) ** 2
        if not np.all(np.isclose(ex_probs, 0, atol=tol)):
            raise TypeError(
                "Can't reduce to chosen basis because the population of a "
                "state to eliminate is above the allowed tolerance."
            )

        return ex_inds

    def _reduce_state_vector_to_basis(self, ex_inds: List[int], normalize: bool) -> qutip.Qobj:
        """Reduces the state vector by eliminating the states at given indices.

        Args:
            ex_inds: The indices of the states to eliminate.

        Returns:
            A Qobj instance representing the state vector with the states eliminated.
        """
        state = self.state.copy()
        state = state.eliminate_states(ex_inds, normalize=normalize)
        return state
    
    def _reduce_density_matrix_to_basis(self, ex_inds: List[int], normalize: bool) -> qutip.Qobj:
        """Reduces the density matrix by tracing out the subsystems at given indices.

        Args:
            ex_inds: The indices of the subsystems to trace out.

        Returns:
            The subsystem state, reduced by tracing out the subsystems at given indices.
        """
        state = self.state.copy()
        reduced_state = state.ptrace(ex_inds)
        
        # Normalize the density matrix by ensuring that its trace equals 1.
        if normalize:
            reduced_state = reduced_state / reduced_state.tr()

        return reduced_state

    def get_state(
        self,
        reduce_to_basis: str | None = None,
        ignore_global_phase: bool = True,
        tol: float = 1e-6,
        normalize: bool = True,
    ) -> qutip.Qobj:
        state = self.state.copy()
        is_density_matrix = state.isoper
        if ignore_global_phase and not is_density_matrix:
            full = state.full()
            global_ph = float(np.angle(full[np.argmax(np.abs(full))]))
            state *= np.exp(-1j * global_ph)
        if self._dim != 3:
            if reduce_to_basis not in [None, self._basis_name]:
                raise TypeError(
                    f"Can't reduce a system in {self._basis_name}"
                    + f" to the {reduce_to_basis} basis."
                )
        elif reduce_to_basis is not None:
            ex_inds = self._get_relevant_subsystem_indices(reduce_to_basis, tol)
            if is_density_matrix:
                state = self._reduce_density_matrix_to_basis(ex_inds, normalize)
            else:
                state = self._reduce_state_vector_to_basis(ex_inds, normalize)
        return state.tidyup()