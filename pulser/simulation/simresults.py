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

from collections import Counter
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import qutip
import numpy as np


class SimulationResults(ABC):
    """Results of a simulation run of a pulse sequence.

    Parent class for NoisyResults and CleanResults.

    Contains methods for studying the states and extracting useful information
    from them.
    """

    def __init__(self, run_output, dim, size, basis_name, meas_basis,
                 sim_times):
        """Initializes a new SimulationResults instance.

        Args:
            run_output (List[qutip.Qobj]): List of ``qutip.Qobj`` corresponding
                to the states at each time step after the evolution has been
                simulated.
            dim (int): The dimension of the local space of each atom (2 or 3).
            size (int): The number of atoms in the register.
            basis_name (str): The basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg', 'digital' or 'all').
            sim_times (array): Array of times when simulation results are
                returned.

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
        self.sim_times = sim_times

    @abstractmethod
    def expect(self, obs_list):
        """Calculates the expectation value of a list of observables.

        Args:
            obs_list (array-like of qutip.Qobj or array-like of numpy.ndarray):
                A list of observables whose expectation value will be
                calculated. If necessary, each member will be transformed into
                a qutip.Qobj instance.
        """
        pass

    @abstractmethod
    def sample_state(self, t=-1, meas_basis=None, N_samples=1000):
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
        pass

    @abstractmethod
    def sample_final_state(self, meas_basis=None, N_samples=1000):
        pass

    @abstractmethod
    def plot(self, op, fmt=''):
        pass


class NoisyResults(SimulationResults):
    """Results of a noisy simulation run of a pulse sequence.

    Contrary to a CleanResults object, this object contains a unique Counter
    describing the state distribution at the time it was created by using
    Simulation.run() with a noisy simulation.

    Contains methods for studying the populations and extracting useful
    information from them.
    """

    def __init__(self, run_output, size, basis_name, meas_basis, sim_times,
                 N_measures, dim=2):
        """Initializes a new NoisyResults instance.

        Warning :
            Can't have single-atom Hilbert spaces with dimension bigger
            than 2 for NoisyResults objects.
            This is not the case for a CleanResults object, containing states
            in Hilbert space, but NoisyResults contains a probability
            distribution of bitstrings, not atomic states

        Args:
            run_output (Counter List): Each Counter contains the
                probability distribution of a multi-qubits state,
                represented as a bitstring. There is one Counter for each time
                the simulation was asked to return a result.
            dim (int): The dimension of the local space of each atom (2 or 3).
            size (int): The number of atoms in the register.
            basis_name (str): The basis indicating the addressed atoms after
                the pulse sequence ('ground-rydberg', 'digital' or 'all').

        Keyword Args:
            meas_basis (None or str): The basis in which a sampling measurement
                is desired.
            N_measures (int): number of measurements needed to compute this
                result when doing the simulation.
        """
        super().__init__(run_output, dim, size, basis_name, meas_basis,
                         sim_times)
        self.N_measures = N_measures

    @property
    def states(self):
        """Probability distribution of the bitstrings"""
        return self._states

    def get_state(self, t):
        """Get the state at time t of the simulation as a diagonal density
        matrix.
        This is not the density matrix of the system, but is a convenient way
        of computing expectation values of observables.

        Args:
            t (int): index of the state to be returned.

        Returns:
            qutip.Qobj: States probability distribution as a density matrix.
        """
        def _proj_from_bitstring(bitstring):
            # In the digital case, |h> = |1> = qutip.basis()
            # 'all' basis is unacceptable here after projection on bitstrings
            if self._meas_basis == 'digital':
                proj = qutip.tensor([qutip.basis(2, int(i)).proj() for i
                                     in bitstring])
            # ground-rydberg measurement basis case
            else:
                proj = qutip.tensor([qutip.basis(2, 1-int(i)).proj() for i
                                     in bitstring])
            return proj

        return sum(v * _proj_from_bitstring(b) for
                   b, v in self._states[t].items())

    def get_final_state(self):
        """Get the final state of the simulation as a diagonal density matrix.
        This is not the density matrix of the system, but is a convenient way
        of computing expectation values of observables.

        Returns:
            qutip.Qobj: States probability distribution as a density matrix.
        """
        return self.get_state(-1)

    def expect(self, obs_list):
        """Calculates the expectation value of a list of observables.

        Args:
            obs_list (array-like of qutip.Qobj or array-like of numpy.ndarray):
                A list of observables whose expectation value will be
                calculated. If necessary, each member will be transformed into
                a qutip.Qobj instance.
        Returns:
            list: the list of expectation values of each operator.
        """
        density_matrices = [self.get_state(t) for
                            t in range(len(self._states))]
        if not isinstance(obs_list, (list, np.ndarray)):
            raise TypeError("`obs_list` must be a list of operators")

        qobj_list = []
        for obs in obs_list:
            if not (isinstance(obs, np.ndarray)
                    or isinstance(obs, qutip.Qobj)):
                raise TypeError("Incompatible type of observable.")
            if obs.shape != (2**self._size, 2**self._size):
                raise ValueError("Incompatible shape of observable.")
            qobj_list.append(qutip.Qobj(obs))

        return qutip.expect(qobj_list, density_matrices)

    def sample_state(self, t=-1, N_samples=1000):
        r"""Returns the result of multiple measurements. No notion of
            measurement basis here, since states have already been projected
            onto bitstrings.
        Keyword Args:
            N_samples (int, default=1000): Number of samples to take.
            t (int, default=-1) : Time at which the system is measured.

        Raises:
            ValueError: If trying to sample without a defined 'meas_basis' in
                the arguments when the original sequence is not measured.
        """
        N = self._size
        self.N_samples = N_samples
        bitstrings = [np.binary_repr(k, N) for k in range(2**N)]
        probs = [self._states[t][b] for b in bitstrings]

        dist = np.random.multinomial(N_samples, probs)
        return Counter(
               {np.binary_repr(i, N): dist[i] for i in np.nonzero(dist)[0]})

    def sample_final_state(self, N_samples=1000):
        return self.sample_state(N_samples=N_samples)

    def _standard_dev(self, op):
        """Returns the square root of the variance of operator op."""
        density_mats = [self.get_state(t) for t in range(len(self._states))]
        return np.sqrt(qutip.variance(op, density_mats) / self.N_measures)

    def _get_error_bars(self, op):
        moy = self.expect([op])[0]
        st = self._standard_dev(op)
        return moy, st

    def plot(self, op, error_bars=True, fmt='.'):
        """Plots the expectation results of operator op, computing error bars
            if wanted.
        Args:
            op (Qobj): QuTiP operator which expectation value is to be plotted.
            error_bars (bool): display error bars or not.
        """
        if error_bars:
            moy, st = self._get_error_bars(op)
            plt.errorbar(self.sim_times, moy, st, fmt=fmt)
        else:
            plt.plot(self.sim_times, self.expect([op])[0], fmt)


class CleanResults(SimulationResults):
    """Results of an ideal simulation run of a pulse sequence.

    Contains methods for studying the states and extracting useful information
    from them.
    """

    def __init__(self, run_output, dim, size, basis_name,
                 meas_basis, sim_times):
        """Initializes a new CleanResults instance.

        Args:
            run_output (list of qutip.Qobj): List of `qutip.Qobj` corresponding
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
        super().__init__(run_output, dim, size, basis_name, meas_basis,
                         sim_times)

    @property
    def states(self):
        """List of ``qutip.Qobj`` for each state in the simulation."""
        return list(self._states)

    def get_final_state(self, reduce_to_basis=None, ignore_global_phase=True,
                        tol=1e-6, normalize=True):
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

    def expect(self, obs_list):
        """Calculates the expectation value of a list of observables.

        Args:
            obs_list (Array[qutip, numpy.ndarray]): A list of observables whose
                expectation value will be calculated. If necessary, each member
                will be transformed into a ``qutip.Qobj`` instance.
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

    def sample_state(self, t=-1, meas_basis=None, N_samples=1000):
        r"""Returns the result of multiple measurements in a given basis.

        The encoding of the results depends on the meaurement basis. Namely:

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
        if not meas_basis:
            meas_basis = self._meas_basis
        if meas_basis not in {'ground-rydberg', 'digital'}:
            raise ValueError(
                "`meas_basis` can only be 'ground-rydberg' or 'digital'."
                )

        N = self._size
        self.N_samples = N_samples
        final_state = self._states[t].unit()
        # Case of a density matrix
        if final_state.type != "ket":
            probs = np.abs(final_state.diag())
        else:
            probs = (np.abs(final_state.full())**2).flatten()

        if self._dim == 2:
            if meas_basis == self._basis_name:
                # State vector ordered with r first for 'ground_rydberg'
                # e.g. N=2: [rr, rg, gr, gg] -> [11, 10, 01, 00]
                # Invert the order ->  [00, 01, 10, 11] correspondence
                weights = probs if meas_basis == 'digital' else probs[::-1]
            else:
                return {'0' * N: int(N_samples)}

        elif self._dim == 3:
            if meas_basis == 'ground-rydberg':
                one_state = 0       # 1 = |r>
                ex_one = slice(1, 3)
            elif meas_basis == 'digital':
                one_state = 2       # 1 = |h>
                ex_one = slice(0, 2)
            probs = probs.reshape([3]*N)
            weights = np.zeros(2**N)
            for dec_val in range(2**N):
                ind = []
                for v in np.binary_repr(dec_val, width=N):
                    if v == '0':
                        ind.append(ex_one)
                    else:
                        ind.append(one_state)
                # Eg: 'digital' basis => |1> = index 2, |0> = index 0, 1 = 0:2
                # p_11010 = sum(probs[2, 2, 0:2, 2, 0:2])
                # We sum all probabilites that correspond to measuring 11010,
                # namely hhghg, hhrhg, hhghr, hhrhr
                weights[dec_val] = np.sum(probs[tuple(ind)])
        else:
            raise NotImplementedError(
                "Cannot sample system with single-atom state vectors of "
                "dimension > 3."
                )
        # Takes care of potential numerical artefacts in case sum(weights) != 1
        weights /= sum(weights)
        dist = np.random.multinomial(N_samples, weights)
        return Counter(
               {np.binary_repr(i, N): dist[i] for i in np.nonzero(dist)[0]})

    def sample_final_state(self, meas_basis=None, N_samples=1000):
        return self.sample_state(-1, meas_basis, N_samples)

    def detection_from_basis_state(self, N_d, shot, spam):
        """Returns the probability distribution of states really detected
            when the simulation detects bitstring shot.

        Args:
            shot (str): binary string of length the number of atoms of the
            simulation.
            N_samples (int): Number of times state has been detected.
            spam (dict): dictionnary gathering the SPAM error
            probabilities.
        """
        n_0 = shot.count('0')
        n_1 = shot.count('1')
        eps = spam['epsilon']
        eps_p = spam['epsilon_prime']
        # Verified
        prob_1_to_0 = eps_p * (1 - eps) ** n_0 * (1 - eps_p) ** (n_1 - 1)
        prob_0_to_1 = eps * (1 - eps) ** (n_0 - 1) * (1 - eps_p) ** n_1
        probs = [int(shot[i]) * prob_1_to_0 + (1 - int(shot[i]))
                 * prob_0_to_1 for i in range(len(shot))]
        probs += [1 - sum(probs)]
        shots = np.random.multinomial(N_d, probs)
        detected_dict = {shot: shots[-1]}

        for i in range(len(shot)):
            if shots[i]:
                detected_dict[shot[:i] + str(1 - int(shot[i])) + shot[i
                              + 1:]] = shots[i]
        return Counter(detected_dict)

    def sampling_with_detection_errors(self, spam, t=-1,
                                       meas_basis=None,
                                       N_samples=1000):
        """Returns the distribution of states really detected instead of
        sampled_state. Doesn't take state preparation errors into account.
        Part of the SPAM implementation.

        Args:
            sampled_state (dict): dictionnary of detected states as binary
            string with their detection number.
            spam (dict): dictionnary gathering the SPAM error
            probabilities.
        """
        sampled_state = self.sample_state(t=t, meas_basis=meas_basis,
                                          N_samples=N_samples)
        detected_sample_dict = Counter()
        for (shot, N_d) in sampled_state.items():
            dict_state = self.detection_from_basis_state(N_d, shot, spam)
            detected_sample_dict += dict_state

        return detected_sample_dict

    def plot(self, op, fmt=''):
        plt.plot(self.sim_times, self.expect([op])[0], fmt)
