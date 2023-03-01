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

import collections.abc
from abc import ABC, abstractmethod
from collections import Counter
from functools import lru_cache
from typing import Mapping, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import qutip
from numpy.typing import ArrayLike
from qutip.piqs import isdiagonal


class SimulationResults(ABC):
    """Results of a simulation run of a pulse sequence.

    Parent class for NoisyResults and CoherentResults.
    Contains methods for studying the states and extracting useful information
    from them.
    """

    def __init__(
        self, size: int, basis_name: str, sim_times: np.ndarray
    ) -> None:
        """Initializes a new SimulationResults instance.

        Args:
            size: The number of atoms in the register.
            basis_name: The basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg', 'digital' or 'all').
            sim_times: Array of times (µs) when simulation results are
                returned.
        """
        self._dim = 3 if basis_name == "all" else 2
        self._size = size
        if basis_name not in {"ground-rydberg", "digital", "all", "XY"}:
            raise ValueError(
                "`basis_name` must be 'ground-rydberg', 'digital', 'all' or "
                "'XY'."
            )
        self._basis_name = basis_name
        self._sim_times = sim_times
        self._results: Union[list[Counter], list[qutip.Qobj]]
        # Use the pseudo-density matrix when calculating expectation values
        self._use_pseudo_dens: bool = False

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
    def _calc_weights(self, t_index: int) -> np.ndarray:
        """Computes the bitstring probabilities for sampled states."""
        pass

    def expect(
        self, obs_list: collections.abc.Sequence[Union[qutip.Qobj, ArrayLike]]
    ) -> list[Union[float, complex, ArrayLike]]:
        """Returns the expectation values of operators in obs_list.

        Args:
            obs_list: Input observable
                list. ArrayLike objects will be converted to qutip.Qobj.

        Returns:
            Expectation values of obs_list.
        """
        if not isinstance(obs_list, (list, np.ndarray)):
            raise TypeError("`obs_list` must be a list of operators.")

        qobj_list = []
        dim = self._dim if not self._use_pseudo_dens else 2
        legal_dims = [[dim] * self._size] * 2
        legal_shape = (dim**self._size, dim**self._size)
        for obs in obs_list:
            if not (
                isinstance(obs, np.ndarray) or isinstance(obs, qutip.Qobj)
            ):
                raise TypeError(
                    f"Incompatible type {type(obs)} of "
                    + "observable. Type must be ArrayLike or "
                    + "qutip.Qobj."
                )
            if obs.shape != legal_shape:
                raise ValueError(
                    "Incompatible shape of observable."
                    + f"Expected {legal_shape}, got {obs.shape}."
                )
            qobj_list.append(qutip.Qobj(obs, dims=legal_dims))
            if self._use_pseudo_dens:
                if not isdiagonal(obs):
                    raise ValueError(f"Observable {obs!r} is non-diagonal.")
                states = [
                    self._calc_pseudo_density(ind)
                    for ind in range(len(self._results))
                ]
            else:
                states = self.states

        return cast(list, qutip.expect(qobj_list, states))

    def sample_state(
        self, t: float, n_samples: int = 1000, t_tol: float = 1.0e-3
    ) -> Counter:
        """Returns the result of multiple measurements at time t.

        Args:
            t: Time at which the state is sampled.
            n_samples: Number of samples to return.
            t_tol: Tolerance for the difference between t and
                closest time.

        Returns:
            Sample distribution of bitstrings corresponding to
            measured quantum states at time t.
        """
        t_index = self._get_index_from_time(t, t_tol)
        dist = np.random.multinomial(n_samples, self._calc_weights(t_index))
        return Counter(
            {
                np.binary_repr(i, self._size): dist[i]
                for i in np.nonzero(dist)[0]
            }
        )

    def sample_final_state(self, N_samples: int = 1000) -> Counter:
        """Returns the result of multiple measurements of the final state.

        Args:
            N_samples: Number of samples to return.

        Returns:
            Sample distribution of bitstrings corresponding to
            measured quantum states at the end of the simulation.
        """
        return self.sample_state(self._sim_times[-1], N_samples)

    def plot(self, op: qutip.Qobj, fmt: str = "", label: str = "") -> None:
        """Plots the expectation value of a given operator op.

        Args:
            op: Operator whose expectation value is wanted.
            fmt: Curve plot format.
            label: Curve label.
        """
        plt.plot(self._sim_times, self.expect([op])[0], fmt, label=label)
        plt.xlabel("Time (µs)")
        plt.ylabel("Expectation value")

    def _get_index_from_time(self, t_float: float, tol: float = 1.0e-3) -> int:
        """Returns closest index corresponding to time t_float.

        Args:
            t_float: Time value (in µs).
            tol: Tolerance for the difference between t_float and
                closest time.
        """
        try:
            return int(np.where(abs(t_float - self._sim_times) < tol)[0][0])
        except IndexError:
            raise IndexError(
                f"Given time {t_float} is absent from Simulation times within"
                + f" tolerance {tol}."
            )

    @lru_cache(maxsize=None)
    def _calc_pseudo_density(self, t_index: int) -> qutip.Qobj:
        """Calculates the pseudo-density matrix at a given time.

        The pseudo-density matrix is the diagonal matrix calculated from the
        probability of obtaining each possible state, after measurement.

        Args:
            t_index: The index in the list of states/results to turn
                into the pseudo-density matrix.

        Returns:
            The pseudo-density matrix as a Qobj.
        """

        def _proj_from_bitstring(bitstring: str) -> qutip.Qobj:
            proj = qutip.tensor(
                [self._meas_projector(int(i)) for i in bitstring]
            )
            return proj

        w = self._calc_weights(t_index)
        return sum(
            w[i] * _proj_from_bitstring(np.binary_repr(i, width=self._size))
            for i in np.nonzero(w)[0]
        )

    def _meas_projector(self, state_n: int) -> qutip.Qobj:
        """Gets the post measurement projector.

        Args:
            state_n: The measured state (0 or 1).
        """
        if self._basis_name == "digital":
            return qutip.basis(2, state_n).proj()

        # 0 = |g or d> = |1>; 1 = |r or u> = |0>
        return qutip.basis(2, 1 - state_n).proj()


class NoisyResults(SimulationResults):
    """Results of a noisy simulation run of a pulse sequence.

    Contrary to a CoherentResults object, this object contains a list of
    Counters describing the state distribution at the time it was created by
    using Simulation.run() with a noisy simulation.
    Contains methods for studying the populations and extracting useful
    information from them.
    """

    def __init__(
        self,
        run_output: list[Counter],
        size: int,
        basis_name: str,
        sim_times: np.ndarray,
        n_measures: int,
    ) -> None:
        """Initializes a new NoisyResults instance.

        Warning:
            Can't have single-atom Hilbert spaces with dimension bigger
            than 2 for NoisyResults objects.
            This is not the case for a CoherentResults object, containing
            states in Hilbert space, but NoisyResults contains a probability
            distribution of bitstrings, not atomic states

        Args:
            run_output: Each Counter contains the
                probability distribution of a multi-qubits state,
                represented as a bitstring. There is one Counter for each time
                the simulation was asked to return a result.
            size: The number of atoms in the register.
            basis_name: Basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg' or 'digital' - 'all' basis
                makes no sense after projection on bitstrings). Defaults to
                'digital' if given value 'all'.
            sim_times: Times at which Simulation object returned
                the results.
            n_measures: Number of measurements needed to compute this
                result when doing the simulation.
        """
        basis_name_ = "digital" if basis_name == "all" else basis_name
        super().__init__(size, basis_name_, sim_times)
        self.n_measures = n_measures
        self._results = run_output
        self._use_pseudo_dens = True

    @property
    def states(self) -> list[qutip.Qobj]:
        """Measured states as a list of diagonal qutip.Qobj."""
        return [self.get_state(t) for t in self._sim_times]

    @property
    def results(self) -> list[Counter]:
        """Probability distribution of the bitstrings."""
        return self._results

    def get_state(self, t: float, t_tol: float = 1.0e-3) -> qutip.Qobj:
        """Gets the state at time t as a diagonal density matrix.

        Note:
            This is not the density matrix of the system, but is a convenient
            way of computing expectation values of observables.

        Args:
            t: Time (µs) at which to return the state.
            t_tol: Tolerance for the difference between t and
                closest time.

        Returns:
            States probability distribution as a diagonal density matrix.
        """
        t_index = self._get_index_from_time(t, t_tol)
        return self._calc_pseudo_density(t_index)

    def get_final_state(self) -> qutip.Qobj:
        """Get the final state of the simulation as a diagonal density matrix.

        Note:
            This is not the density matrix of the system, but is a convenient
            way of computing expectation values of observables.

        Returns:
            States probability distribution as a density matrix.
        """
        return self.get_state(self._sim_times[-1])

    def _calc_weights(self, t_index: int) -> np.ndarray:
        weights = np.zeros(2**self._size)
        for bin_rep, prob in self._results[t_index].items():
            weights[int(bin_rep, base=2)] = prob
        return weights

    def plot(
        self,
        op: qutip.Qobj,
        fmt: str = ".",
        label: str = "",
        error_bars: bool = True,
    ) -> None:
        """Plots the expectation value of a given operator op.

        Note:
            The observable must be diagonal.

        Args:
            op: Operator whose expectation value is wanted.
            fmt: Curve plot format.
            label: y-Axis label.
            error_bars: Choose to display error bars.
        """

        def get_error_bars() -> Tuple[ArrayLike, ArrayLike]:
            moy = self.expect([op])[0]
            standard_dev = cast(
                np.ndarray,
                np.sqrt(qutip.variance(op, self.states) / self.n_measures),
            )
            return moy, standard_dev

        if error_bars:
            moy, st = get_error_bars()
            plt.errorbar(
                self._sim_times, moy, st, fmt=fmt, lw=1, capsize=3, label=label
            )
            plt.xlabel("Time (µs)")
            plt.ylabel("Expectation value")
        else:
            super().plot(op, fmt, label)


class CoherentResults(SimulationResults):
    """Results of a coherent simulation run of a pulse sequence.

    Contains methods for studying the states and extracting useful information
    from them.
    """

    def __init__(
        self,
        run_output: list[qutip.Qobj],
        size: int,
        basis_name: str,
        sim_times: np.ndarray,
        meas_basis: str,
        meas_errors: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Initializes a new CoherentResults instance.

        Args:
            run_output: List of `qutip.Qobj` corresponding
                to the states at each time step after the evolution has been
                simulated.
            size: The number of atoms in the register.
            basis_name: The basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg', 'digital' or 'all').
            sim_times: Times at which Simulation object returned the
                results.
            meas_basis: The basis in which a sampling measurement
                is desired.
            meas_errors: If measurement errors
                are involved, give them in a dictionary with "epsilon" and
                "epsilon_prime".
        """
        super().__init__(size, basis_name, sim_times)
        if self._basis_name == "all":
            if meas_basis not in {"ground-rydberg", "digital"}:
                raise ValueError(
                    "`meas_basis` must be 'ground-rydberg' or 'digital'."
                )
        else:
            if meas_basis != self._basis_name:
                raise ValueError(
                    "`meas_basis` and `basis_name` must have the same value."
                )
        self._meas_basis = meas_basis
        self._results = run_output
        if meas_errors is not None:
            if set(meas_errors) != {"epsilon", "epsilon_prime"}:
                raise ValueError(
                    "When defining measurement errors, only values of "
                    "'epsilon' and 'epsilon_prime' must be given."
                )
            self._use_pseudo_dens = True
        self._meas_errors = meas_errors

    @property
    def states(self) -> list[qutip.Qobj]:
        """List of ``qutip.Qobj`` for each state in the simulation."""
        return list(self._results)

    def get_state(
        self,
        t: float,
        reduce_to_basis: Optional[str] = None,
        ignore_global_phase: bool = True,
        tol: float = 1e-6,
        normalize: bool = True,
        t_tol: float = 1.0e-3,
    ) -> qutip.Qobj:
        """Get the state at time t of the simulation.

        Args:
            t: Time (µs) at which to return the state.
            reduce_to_basis: Reduces the full state vector
                to the given basis ("ground-rydberg" or "digital"), if the
                population of the states to be ignored is negligible. Doesn't
                apply to XY mode.
            ignore_global_phase: If True, changes the
                final state's global phase such that the largest term (in
                absolute value) is real.
            tol: Maximum allowed population of each
                eliminated state.
            normalize: Whether to normalize the reduced
                state.
            t_tol: Tolerance for the difference between t and
                closest time.

        Returns:
            The resulting state at time t.

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
                raise TypeError(
                    f"Can't reduce a system in {self._basis_name}"
                    + f" to the {reduce_to_basis} basis."
                )
        elif reduce_to_basis is not None:
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

    def get_final_state(
        self,
        reduce_to_basis: Optional[str] = None,
        ignore_global_phase: bool = True,
        tol: float = 1e-6,
        normalize: bool = True,
    ) -> qutip.Qobj:
        """Returns the final state of the Simulation.

        Args:
            reduce_to_basis: Reduces the full state vector
                to the given basis ("ground-rydberg" or "digital"), if the
                population of the states to be ignored is negligible. Doesn't
                apply to XY mode.
            ignore_global_phase: If True, changes the
                final state's global phase such that the largest term (in
                absolute value) is real.
            tol: Maximum allowed population of each
                eliminated state.
            normalize: Whether to normalize the reduced
                state.

        Returns:
            The resulting final state.

        Raises:
            If trying to reduce to a basis that would eliminate states with
            significant occupation probabilites.
        """
        return self.get_state(
            self._sim_times[-1],
            reduce_to_basis,
            ignore_global_phase,
            tol,
            normalize,
        )

    def _calc_weights(self, t_index: int) -> np.ndarray:
        n = self._size
        state_t = cast(qutip.Qobj, self._results[t_index]).unit()
        if state_t.type != "ket":
            probs = np.abs(state_t.diag())
        else:
            probs = (np.abs(state_t.full()) ** 2).flatten()

        if self._dim == 2:
            if self._meas_basis == self._basis_name:
                # State vector ordered with r first for 'ground_rydberg'
                # e.g. n=2: [rr, rg, gr, gg] -> [11, 10, 01, 00]
                # Invert the order ->  [00, 01, 10, 11] correspondence
                # The same applies in XY mode, which is ordered with u first
                weights = (
                    probs if self._meas_basis == "digital" else probs[::-1]
                )
            else:
                # Only 000...000 is measured
                weights = np.zeros(probs.size)
                weights[0] = 1.0

        elif self._dim == 3:
            if self._meas_basis == "ground-rydberg":
                one_state = 0  # 1 = |r>
                ex_one = slice(1, 3)
            elif self._meas_basis == "digital":
                one_state = 2  # 1 = |h>
                ex_one = slice(0, 2)
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
        weights /= sum(weights)
        return cast(np.ndarray, weights)

    def _meas_projector(self, state_n: int) -> qutip.Qobj:
        if self._meas_errors:
            err_param = (
                self._meas_errors["epsilon"]
                if state_n == 0
                else self._meas_errors["epsilon_prime"]
            )
            # 'good' is the position of the state that measures to state_n
            # Matches for the digital basis, is inverted for ground-rydberg and
            # for XY
            good = state_n if self._basis_name == "digital" else 1 - state_n
            return (
                qutip.basis(2, good).proj() * (1 - err_param)
                + qutip.basis(2, 1 - good).proj() * err_param
            )
        # Returns normal projectors in the absence of measurement errors
        return super()._meas_projector(state_n)

    def sample_state(
        self, t: float, n_samples: int = 1000, t_tol: float = 1.0e-3
    ) -> Counter:
        """Returns the result of multiple measurements at time t.

        Args:
            t: Time at which the state is sampled.
            n_samples: Number of samples to return.
            t_tol: Tolerance for the difference between t and
                closest time.

        Returns:
            Sample distribution of bitstrings corresponding to measured
            quantum states at time t.
        """
        sampled_state = super().sample_state(t, n_samples, t_tol)
        if self._meas_errors is None:
            return sampled_state

        detected_sample_dict: Counter = Counter()
        for shot, n_detects in sampled_state.items():
            eps = self._meas_errors["epsilon"]
            eps_p = self._meas_errors["epsilon_prime"]
            # Shot as an array of 1s and 0s
            shot_arr = np.array(list(shot), dtype=int)
            # Probability of flipping each bit
            flip_probs = np.array([eps_p if x == "1" else eps for x in shot])
            # 1 if it flips, 0 if it stays the same
            flips = (
                np.random.uniform(size=(n_detects, len(flip_probs)))
                < flip_probs
            ).astype(int)
            # XOR betwen the original array and the flips
            # Gives an array of n_detects individual shots
            new_shots = shot_arr ^ flips
            # Count all the new_shots
            detected_sample_dict += Counter(
                "".join(map(str, measured)) for measured in new_shots
            )

        return detected_sample_dict
