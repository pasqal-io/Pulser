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
from typing import Union, cast

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

    def get_state(
        self,
        reduce_to_basis: str | None = None,
        ignore_global_phase: bool = True,
        tol: float = 1e-6,
        normalize: bool = True,
    ) -> qutip.Qobj:
        """Gets the state with some optional post-processing.

        Args:
            reduce_to_basis: Reduces the full state vector
                to the given basis ("ground-rydberg" or "digital"), if the
                population of the states to be ignored is negligible. Doesn't
                apply to XY mode.
            ignore_global_phase: If True and if the final state is a vector,
                changes the final state's global phase such that the largest
                term (in absolute value) is real.
            tol: Maximum allowed population of each eliminated state.
            normalize: Whether to normalize the reduced state.

        Returns:
            The resulting state.

        Raises:
            TypeError: If trying to reduce to a basis that would eliminate
                states with significant occupation probabilites.
        """
        state = self.state.copy()
        is_density_matrix = state.isoper
        if ignore_global_phase and not is_density_matrix:
            full = state.full()
            global_ph = float(np.angle(full[np.argmax(np.abs(full))])[0])
            state *= np.exp(-1j * global_ph)
        if self._dim != 3:
            if reduce_to_basis not in [None, self._basis_name]:
                raise TypeError(
                    f"Can't reduce a system in {self._basis_name}"
                    + f" to the {reduce_to_basis} basis."
                )
        elif reduce_to_basis is not None:
            if is_density_matrix:  # pragma: no cover
                # Not tested as noise in digital or all basis not implemented
                raise NotImplementedError(
                    "Reduce to basis not implemented for density matrix"
                    " states."
                )
            if reduce_to_basis == "ground-rydberg":
                ex_state = "2"
            elif reduce_to_basis == "digital":
                ex_state = "0"
            else:
                raise ValueError(
                    "'reduce_to_basis' must be 'ground-rydberg' "
                    + f"or 'digital', not '{reduce_to_basis}'."
                )
            ex_inds = [
                i
                for i in range(3**self._size)
                if ex_state in np.base_repr(i, base=3).zfill(self._size)
            ]
            ex_probs = np.abs(state.extract_states(ex_inds).full()) ** 2
            if not np.all(np.isclose(ex_probs, 0, atol=tol)):
                raise TypeError(
                    "Can't reduce to chosen basis because the population of a "
                    "state to eliminate is above the allowed tolerance."
                )
            state = state.eliminate_states(ex_inds, normalize=normalize)
        return state.tidyup()
