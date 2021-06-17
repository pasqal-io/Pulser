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
"""Classes for containing and processing the results of a simulation."""

from __future__ import annotations
from collections import Counter
from abc import ABC, abstractmethod
from typing import Optional, Union, cast, Tuple
from collections.abc import Sequence

import matplotlib.pyplot as plt
import qutip
from qutip.piqs import isdiagonal
import numpy as np
from numpy.typing import ArrayLike


class SimulationResults(ABC):
    """Results of a simulation run of a pulse sequence.

    Parent class for NoisyResults and CleanResults.
    Contains methods for studying the states and extracting useful information
    from them.
    """

    def __init__(self, size: int, basis_name: str,
                 sim_times: np.ndarray) -> None:
        """Initializes a new SimulationResults instance.

        Args:
            size (int): The number of atoms in the register.
            basis_name (str): The basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg', 'digital' or 'all').
            sim_times (array): Array of times (µs) when simulation results are
                returned.
        """
        self._dim = 3 if basis_name == "all" else 2
        self._size = size
        if basis_name not in {'ground-rydberg', 'digital', 'all'}:
            raise ValueError(
                "`basis_name` must be 'ground-rydberg', 'digital' or 'all'."
                )
        self._basis_name = basis_name
        self._sim_times = sim_times
        self._results: Union[list[Counter], list[qutip.Qobj]]

    @property
    @abstractmethod
    def states(self) -> list[qutip.Qobj]:
        """Lists states of the system at simulation times."""
        pass

    @abstractmethod
    def get_state(self, t: float) -> qutip.Qobj:
        """Returns the state of the system at time t."""
        pass

    @abstractmethod
    def get_final_state(self) -> qutip.Qobj:
        """Returns the final state of the system."""
        pass

    @abstractmethod
    def _calc_weights(self, t_index: int) -> ArrayLike:
        """Computes the bitstring probabilities for sampled states."""
        pass

    def expect(self, obs_list: Sequence[Union[qutip.Qobj, ArrayLike]]
               ) -> list[Union[float, complex, ArrayLike]]:
        """Returns the expectation values of operators in obs_list."""
        if not isinstance(obs_list, (list, np.ndarray)):
            raise TypeError("`obs_list` must be a list of operators.")

        qobj_list = []
        for obs in obs_list:
            if not (isinstance(obs, np.ndarray)
                    or isinstance(obs, qutip.Qobj)):
                raise TypeError("Incompatible type of observable.")
            if obs.shape != (2**self._size, 2**self._size):
                raise ValueError("Incompatible shape of observable.")
            qobj_list.append(qutip.Qobj(obs))

        return cast(list, qutip.expect(qobj_list, self.states))

    def sample_state(self, t: float, n_samples: int = 1000,
                     t_tol: float = 1.e-3) -> Counter:
        """Returns the result of multiple measurements at time t.

        Args:
            t (float): Time at which the state is sampled.
            n_samples (int): Number of samples to return.
            t_tol (float): Tolerance for the difference between t and
                closest time.
        """
        t_index = self._get_index_from_time(t, t_tol)
        dist = np.random.multinomial(n_samples, self._calc_weights(t_index))
        return Counter({np.binary_repr(
                        i, self._size): dist[i] for i in np.nonzero(dist)[0]})

    def sample_final_state(self, n_samples: int = 1000) -> Counter:
        """Returns the result of multiple measurements of the final state."""
        return self.sample_state(self._sim_times[-1], n_samples)

    def plot(self, op: qutip.Qobj, fmt: str = '', label: str = '') -> None:
        """Plots the expectation value of a given operator op.

        Args:
            op (qutip.Qobj): Operator whose expectation value is wanted.
            fmt (str): Curve plot format.
            label (str): Axis label.
        """
        plt.plot(self._sim_times, self.expect([op])[0], fmt, label=label)
        plt.xlabel('Time (µs)')
        plt.ylabel('Expectation value')

    def _get_index_from_time(self, t_float: float, tol: float = 1.e-3) -> int:
        """Returns closest index corresponding to time t_float.

        Args:
            t_float (float): Time the time index of which is needed.
            tol (float): Tolerance for the difference between t_float and
                closest time.
        """
        try:
            return int(np.where(abs(t_float - self._sim_times) < tol)[0][0])
        except IndexError:
            raise IndexError(
                f"Given time {t_float} is absent from Simulation times within"
                + f" tolerance {tol}.")


class NoisyResults(SimulationResults):
    """Results of a noisy simulation run of a pulse sequence.

    Contrary to a CleanResults object, this object contains a list of Counter
    describing the state distribution at the time it was created by using
    Simulation.run() with a noisy simulation.
    Contains methods for studying the populations and extracting useful
    information from them.
    """

    def __init__(self, run_output: list[Counter],
                 size: int, basis_name: str,
                 sim_times: np.ndarray, n_measures: int) -> None:
        """Initializes a new NoisyResults instance.

        Warning:
            Can't have single-atom Hilbert spaces with dimension bigger
            than 2 for NoisyResults objects.
            This is not the case for a CleanResults object, containing states
            in Hilbert space, but NoisyResults contains a probability
            distribution of bitstrings, not atomic states

        Args:
            run_output (list[Counter]): Each Counter contains the
                probability distribution of a multi-qubits state,
                represented as a bitstring. There is one Counter for each time
                the simulation was asked to return a result.
            size (int): The number of atoms in the register.
            basis_name (str): Basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg' or 'digital' - 'all' basis
                makes no sense after projection on bitstrings).
            sim_times (list): Times at which Simulation object returned the
                results.
            meas_basis (Optional[str]): The basis in which a sampling
                measurement is desired.
            n_measures (int): Number of measurements needed to compute this
                result when doing the simulation.
        """
        if basis_name not in {'ground-rydberg', 'digital'}:
            raise ValueError("`basis_name` must be either 'ground-rydberg' or"
                             + " 'digital'.")
        super().__init__(size, basis_name, sim_times)
        self.n_measures = n_measures
        self._results = run_output

    @property
    def states(self) -> list[qutip.Qobj]:
        """Measured states as a list of diagonal qutip.Qobj."""
        return [self.get_state(t) for t in self._sim_times]

    @property
    def results(self) -> list[Counter]:
        """Probability distribution of the bitstrings."""
        return self._results

    def get_state(self, t: float, t_tol: float = 1.e-3) -> qutip.Qobj:
        """Get the state at time t as a diagonal density matrix.

        Note:
            This is not the density matrix of the system, but is a convenient
            way of computing expectation values of observables.

        Args:
            t (float): Time (µs) at which to return the state.
            t_tol (float): Tolerance for the difference between t and
                closest time.

        Returns:
            qutip.Qobj: States probability distribution as a diagonal
                density matrix.
        """
        def _proj_from_bitstring(bitstring: str) -> qutip.Qobj:
            # In the digital case, |h> = |1> = qutip.basis()
            if self._basis_name == 'digital':
                proj = qutip.tensor([qutip.basis(2, int(i)).proj() for i
                                     in bitstring])
            # ground-rydberg basis case
            else:
                proj = qutip.tensor([qutip.basis(2, 1-int(i)).proj() for i
                                     in bitstring])
            return proj

        t_index = self._get_index_from_time(t, t_tol)
        return sum(v * _proj_from_bitstring(b) for
                   b, v in self._results[t_index].items())

    def get_final_state(self) -> qutip.Qobj:
        """Get the final state of the simulation as a diagonal density matrix.

        Note: This is not the density matrix of the system, but is a convenient
            way of computing expectation values of observables.

        Returns:
            qutip.Qobj: States probability distribution as a density matrix.
        """
        return self.get_state(self._sim_times[-1])

    def expect(self, obs_list: Sequence[Union[qutip.Qobj, ArrayLike]]
               ) -> list[Union[float, complex, ArrayLike]]:
        """Calculates the expectation value of a list of observables.

        Args:
            obs_list (array-like of qutip.Qobj or array-like of numpy.ndarray):
                A list of observables whose expectation value will be
                calculated. If necessary, each member will be transformed into
                a qutip.Qobj instance.

        Note: This only works for diagonal observables, since results have been
            projected onto the Z basis.

        Returns:
            list: List of expectation values of each operator.
        """
        for obs in obs_list:
            if not isdiagonal(obs):
                raise ValueError(f"Observable {obs} is non-diagonal.")

        return super().expect(obs_list)

    def _calc_weights(self, t_index: int) -> list[float]:
        n = self._size
        bitstrings = [np.binary_repr(k, n) for k in range(2**n)]
        return [self._results[t_index][b] for b in bitstrings]

    def plot(self, op: qutip.Qobj, fmt: str = '.',
             label: str = '', error_bars: bool = True) -> None:
        """Plots the expectation value of a given operator op.

        Note: The observable must be diagonal.

        Args:
            op (qutip.Qobj): Operator whose expectation value is wanted.
            fmt (str): Curve plot format.
            label (str): y-Axis label.
            error_bars (bool): Choose to display error bars.
        """
        def get_error_bars() -> Tuple[ArrayLike, ArrayLike]:
            moy = self.expect([op])[0]
            standard_dev = cast(np.ndarray, np.sqrt(
                qutip.variance(op, self.states) / self.n_measures))
            return moy, standard_dev

        if error_bars:
            moy, st = get_error_bars()
            plt.errorbar(self._sim_times, moy, st, fmt=fmt, lw=1, capsize=3,
                         label=label)
            plt.xlabel('Time (µs)')
            plt.ylabel('Expectation value')
        else:
            super().plot(op, fmt, label)


class CleanResults(SimulationResults):
    """Results of an ideal simulation run of a pulse sequence.

    Contains methods for studying the states and extracting useful information
    from them.
    """

    def __init__(self, run_output: list[qutip.Qobj],
                 size: int, basis_name: str,
                 sim_times: np.ndarray, meas_basis: str) -> None:
        """Initializes a new CleanResults instance.

        Args:
            run_output (list of qutip.Qobj): List of `qutip.Qobj` corresponding
                to the states at each time step after the evolution has been
                simulated.
            size (int): The number of atoms in the register.
            basis_name (str): The basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg', 'digital' or 'all').
            sim_times (list): Times at which Simulation object returned the
                results.
            meas_basis (str): The basis in which a sampling measurement
                is desired.
        """
        super().__init__(size, basis_name, sim_times)
        if meas_basis:
            if meas_basis not in {'ground-rydberg', 'digital'}:
                raise ValueError(
                    "`meas_basis` must be 'ground-rydberg' or 'digital'.")
        self._meas_basis = meas_basis
        self._results = run_output

    @property
    def states(self) -> list[qutip.Qobj]:
        """List of ``qutip.Qobj`` for each state in the simulation."""
        return list(self._results)

    def get_state(self, t: float, reduce_to_basis: Optional[str] = None,
                  ignore_global_phase: bool = True, tol: float = 1e-6,
                  normalize: bool = True, t_tol: float = 1.e-3) -> qutip.Qobj:
        """Get the state at time t of the simulation.

        Args:
            t (float): Time (µs) at which to return the state.
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
            t_tol (float): Tolerance for the difference between t and
                closest time.

        Returns:
            qutip.Qobj: The resulting final state.

        Raises:
            TypeError: If trying to reduce to a basis that would eliminate
                states with significant occupation probabilites.
        """
        t_index = self._get_index_from_time(t, t_tol)
        state = cast(qutip.Qobj, self._results[t_index].copy())
        if ignore_global_phase:
            full = state.full()
            global_ph = float(np.angle(full[np.argmax(np.abs(full))]))
            state *= np.exp(-1j * global_ph)
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
            ex_probs = np.abs(state.extract_states(ex_inds).full()) ** 2
            if not np.all(np.isclose(ex_probs, 0, atol=tol)):
                raise TypeError(
                    "Can't reduce to chosen basis because the population of a "
                    "state to eliminate is above the allowed tolerance."
                    )
            state = state.eliminate_states(ex_inds, normalize=normalize)
        return state.tidyup()

    def get_final_state(self, reduce_to_basis: Optional[str] = None,
                        ignore_global_phase: bool = True, tol: float = 1e-6,
                        normalize: bool = True) -> qutip.Qobj:
        """Returns the final state of the Simulation."""
        return self.get_state(self._sim_times[-1], reduce_to_basis,
                              ignore_global_phase, tol, normalize)

    def _calc_weights(self, t_index: int) -> np.ndarray:
        n = self._size
        state_t = cast(qutip.Qobj, self._results[t_index]).unit()
        # Case of a density matrix
        if state_t.type != "ket":
            probs = np.abs(state_t.diag())
        else:
            probs = (np.abs(state_t.full())**2).flatten()

        if self._dim == 2:
            if self._meas_basis == self._basis_name:
                # State vector ordered with r first for 'ground_rydberg'
                # e.g. n=2: [rr, rg, gr, gg] -> [11, 10, 01, 00]
                # Invert the order ->  [00, 01, 10, 11] correspondence
                weights = (probs if self._meas_basis == 'digital'
                           else probs[::-1])
            else:
                # Only 000...000 is measured
                weights = np.zeros(probs.size)
                weights[0] = 1.

        elif self._dim == 3:
            if self._meas_basis == 'ground-rydberg':
                one_state = 0       # 1 = |r>
                ex_one = slice(1, 3)
            elif self._meas_basis == 'digital':
                one_state = 2       # 1 = |h>
                ex_one = slice(0, 2)
            probs = probs.reshape([3]*n)
            weights = np.zeros(2**n)
            for dec_val in range(2**n):
                ind: list[Union[int, slice]] = []
                for v in np.binary_repr(dec_val, width=n):
                    if v == '0':
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
                "dimension > 3.")
        # Takes care of numerical artefacts in case sum(weights) != 1
        weights /= sum(weights)
        return cast(np.ndarray, weights)

    def _sampling_with_detection_errors(self, spam: dict[str, float],
                                        t: float, n_samples: int = 1000,
                                        ) -> Counter:
        """Returns the distribution of states really detected.

        Doesn't take state preparation errors into account.
        Part of the SPAM implementation.

        Args:
            spam (dict): Dictionnary gathering the SPAM error
            probabilities.
            t (float): Time at which to return the samples.
            n_samples (int): Number of samples.
            t_tol (float): Tolerance for the difference between t and
                closest time.
        """

        def detection_from_basis_state(n_detects: int, shot: str) -> Counter:
            """Returns distribution of states detected when detecting `shot`.

            Part of the SPAM implementation : computes measurement errors.

            Args:
                n_detects (int): Number of times state has been detected.
                shot (str): Binary string of length the number of atoms of the
                simulation.
            """
            n_0 = shot.count('0')
            n_1 = shot.count('1')
            eps = spam['epsilon']
            eps_p = spam['epsilon_prime']

            prob_1_to_0 = eps_p * (1 - eps) ** n_0 * (1 - eps_p) ** (n_1 - 1)
            prob_0_to_1 = eps * (1 - eps) ** (n_0 - 1) * (1 - eps_p) ** n_1
            probs = [prob_1_to_0 if i == '1' else prob_0_to_1 for i in shot]
            probs.append(1. - sum(probs))
            shots = np.random.multinomial(n_detects, probs)
            # Last bitstring : no measurement error, no character flipped
            detected_dict = {shot: shots[-1]}
            for i in range(len(shot)):
                if shots[i]:
                    # Bitstrings equal to shot except for the i-th character
                    # have been detected due to measurement errors :
                    flip_bit_string = (shot[:i] + str(1 - int(shot[i])) +
                                       shot[i+1:])
                    detected_dict[flip_bit_string] = shots[i]
            return Counter(detected_dict)

        sampled_state = self.sample_state(t, n_samples)
        detected_sample_dict: Counter = Counter()
        for (shot, n_d) in sampled_state.items():
            dict_state = detection_from_basis_state(n_d, shot)
            detected_sample_dict += dict_state

        return detected_sample_dict
