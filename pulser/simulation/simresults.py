# Copyright 2020 Pulser Development Team
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

from __future__ import annotations
from typing import Optional, List, Union, Dict, cast

import qutip
import numpy as np
from numpy.typing import ArrayLike


class SimulationResults:
    """Results of a simulation run of a pulse sequence.

    Contains methods for studying the states and extracting useful information
    from them.
    """

    def __init__(self, run_output: List[qutip.Qobj], dim: int, size: int,
                 basis_name: str, meas_basis: Optional[str] = None) -> None:
        """Initializes a new SimulationResults instance.

        Args:
            run_output (List[qutip.Qobj]): List of ``qutip.Qobj`` corresponding
                to the states at each time step after the evolution has been
                simulated.
            dim (int): The dimension of the local space of each atom (2 or 3).
            size (int): The number of atoms in the register.
            basis_name (str): The basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg', 'digital' or 'all').

        Keyword Args:
            meas_basis (None or str): The basis in which a sampling measurement
                is desired.
        """
        self._states = run_output
        self._dim = dim
        self._size = size
        if basis_name not in {'ground-rydberg', 'digital', 'all'}:
            raise ValueError(
                "`basis_name` must be 'ground-rydberg', 'digital' or 'all'."
                )
        self._basis_name = basis_name
        if meas_basis:
            if meas_basis not in {'ground-rydberg', 'digital'}:
                raise ValueError(
                    "`meas_basis` must be 'ground-rydberg' or 'digital'."
                    )
        self._meas_basis = meas_basis

    @property
    def states(self) -> List[List[qutip.Qobj]]:
        """List of ``qutip.Qobj`` for each state in the simulation."""
        return list(self._states)

    def get_final_state(self, reduce_to_basis: Optional[str] = None,
                        ignore_global_phase: bool = True, tol: float = 1e-6,
                        normalize: bool = True) -> qutip.Qobj:
        """Get the final state of the simulation.

        Keyword Args:
            reduce_to_basis (str, default=None): Reduces the full state vector
                to the given basis ("ground-rydberg" or "digital"), if the
                population of the states to be ignored is negligible.
            ignore_global_phase (bool, default=True): If True, changes the
                final state's global phase such that the largest term (in
                absolute value) is real.
            tol (float, default=1e-6): Maximum allowed population of each
                eliminated state.
            normalize (bool, default=True): Whether to normalize the reduced
                state.

        Returns:
            qutip.Qobj: The resulting final state.

        Raises:
            TypeError: If trying to reduce to a basis that would eliminate
                states with significant occupation probabilites.
        """
        final_state = self._states[-1].copy()
        if ignore_global_phase:
            full = final_state.full()
            global_ph = float(np.angle(full[np.argmax(np.abs(full))]))
            final_state *= np.exp(-1j * global_ph)
        if self._dim != 3:
            if reduce_to_basis not in [None, self._basis_name]:
                raise TypeError(f"Can't reduce a system in {self._basis_name}"
                                + f" to the {reduce_to_basis} basis.")
        elif reduce_to_basis is not None:
            if reduce_to_basis == "ground-rydberg":
                ex_state = "2"
            elif reduce_to_basis == "digital":
                ex_state = "0"
            else:
                raise ValueError("'reduce_to_basis' must be 'ground-rydberg' "
                                 + f"or 'digital', not '{reduce_to_basis}'.")
            ex_inds = [i for i in range(3**self._size) if ex_state in
                       np.base_repr(i, base=3).zfill(self._size)]
            ex_probs = np.abs(final_state.extract_states(ex_inds).full()) ** 2
            if not np.all(np.isclose(ex_probs, 0, atol=tol)):
                raise TypeError(
                    "Can't reduce to chosen basis because the population of a "
                    "state to eliminate is above the allowed tolerance."
                    )
            final_state = final_state.eliminate_states(
                                                ex_inds, normalize=normalize)

        return final_state.tidyup()

    def expect(self, obs_list: List[Union[qutip, np.ndarray]]
               ) -> Union[ArrayLike, float]:
        """Calculates the expectation value of a list of observables.

        Args:
            obs_list (List[Union[qutip, np.ndarray]]): A list of observables
                whose expectation value will be calculated. If necessary, each
                member will be transformed into a ``qutip.Qobj`` instance.
        """
        if not isinstance(obs_list, (list, np.ndarray)):
            raise TypeError("`obs_list` must be a list of operators")

        qobj_list = []
        for obs in obs_list:
            if not (isinstance(obs, np.ndarray)
                    or isinstance(obs, qutip.Qobj)):
                raise TypeError("Incompatible type of observable.")
            if obs.shape != (self._dim**self._size, self._dim**self._size):
                raise ValueError("Incompatible shape of observable.")
            # Transfrom to qutip.Qobj and take dims from state
            dim_list = [self._states[0].dims[0], self._states[0].dims[0]]
            qobj_list.append(qutip.Qobj(obs, dims=dim_list))

        return [qutip.expect(qobj, self._states) for qobj in qobj_list]

    def sample_final_state(self, meas_basis: Optional[str] = None,
                           N_samples: int = 1000) -> Dict[str, int]:
        r"""Returns the result of multiple measurements in a given basis.

        The enconding of the results depends on the meaurement basis. Namely:

        - *ground-rydberg* : :math:`1 = |r\rangle;~ 0 = |g\rangle, |h\rangle`
        - *digital* : :math:`1 = |h\rangle;~ 0 = |g\rangle, |r\rangle`

        Note:
            The results are presented using a big-endian representation,
            according to the pre-established qubit ordering in the register.
            This means that, when sampling a register with qubits ('q0','q1',
            ...), in this order, the corresponding value, in binary, will be
            0Bb0b1..., where b0 is the outcome of measuring 'q0', 'b1' of
            measuring 'q1' and so on.

        Keyword Args:
            meas_basis (str, default=None): 'ground-rydberg' or 'digital'. If
                left as None, uses the measurement basis defined in the
                original sequence.
            N_samples (int, default=1000): Number of samples to take.

        Raises:
            ValueError: If trying to sample without a defined 'meas_basis' in
                the arguments when the original sequence is not measured.
        """
        if meas_basis is None:
            if self._meas_basis is None:
                raise ValueError(
                    "Can't accept an undefined measurement basis because the "
                    "original sequence has no measurement."
                    )
            meas_basis = self._meas_basis

        if meas_basis not in {'ground-rydberg', 'digital'}:
            raise ValueError(
                "'meas_basis' can only be 'ground-rydberg' or 'digital'."
                )

        N = self._size
        self.N_samples = N_samples
        probs = np.abs(self._states[-1].full())**2
        if self._dim == 2:
            if meas_basis == self._basis_name:
                # State vector ordered with r first for 'ground_rydberg'
                # e.g. N=2: [rr, rg, gr, gg] -> [11, 10, 01, 00]
                # Invert the order ->  [00, 01, 10, 11] correspondence
                weights = probs if meas_basis == 'digital' else probs[::-1]
            else:
                return {'0' * N: int(N_samples)}
            weights = weights.flatten()

        elif self._dim == 3:
            if meas_basis == 'ground-rydberg':
                one_state = 0       # 1 = |r>
                ex_one = slice(1, 3)
            elif meas_basis == 'digital':
                one_state = 2       # 1 = |h>
                ex_one = slice(0, 2)

            probs = probs.reshape([3]*N)
            weights = []
            for dec_val in range(2**N):
                ind = []
                for v in np.binary_repr(dec_val, width=N):
                    if v == '0':
                        ind.append(ex_one)
                    else:
                        ind.append(cast(slice, one_state))
                # Eg: 'digital' basis => |1> = index 2, |0> = index 0, 1 = 0:2
                # p_11010 = sum(probs[2, 2, 0:2, 2, 0:2])
                # We sum all probabilites that correspond to measuring 11010,
                # namely hhghg, hhrhg, hhghr, hhrhr
                weights.append(np.sum(probs[tuple(ind)]))
        else:
            raise NotImplementedError(
                "Cannot sample system with single-atom state vectors of "
                "dimension > 3."
                )
        dist = np.random.multinomial(N_samples, weights)
        return {np.binary_repr(i, N): dist[i] for i in np.nonzero(dist)[0]}
