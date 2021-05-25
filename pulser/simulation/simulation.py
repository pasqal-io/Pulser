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

import itertools
from collections import Counter
from copy import deepcopy

import qutip
import numpy as np
from pulser import Pulse, Sequence
from pulser.simulation.simresults import CleanResults, NoisyResults
from pulser._seq_drawer import draw_sequence


class Simulation:
    """Simulation of a pulse sequence using QuTiP.

    Creates a Hamiltonian object with the proper dimension according to the
    pulse sequence given, then provides a method to time-evolve an initial
    state using the QuTiP solvers.

    Args:
        sequence (Sequence): An instance of a Pulser Sequence that we
            want to simulate.

    Keyword Args:
        sampling_rate (float): The fraction of samples that we wish to
            extract from the pulse sequence to simulate. Has to be a
            value between 0.05 and 1.0
        evaluation_times (str,list): The list of times at which the quantum
            state should be evaluated, in μs. If 'Full' is provided, this list
            is set to be the one used to define the Hamiltonian to the solver.
            The initial and final times are always included, so that if
            'Minimal' is provided, the list is set to only contain the initial
            and the final times.
    """

    def __init__(self, sequence, sampling_rate=1.0, evaluation_times='Full'):
        """Initialize the Simulation with a specific pulser.Sequence."""
        if not isinstance(sequence, Sequence):
            raise TypeError("The provided sequence has to be a valid "
                            "pulser.Sequence instance.")
        if not sequence._schedule:
            raise ValueError("The provided sequence has no declared channels.")
        if all(sequence._schedule[x][-1].tf == 0 for x in sequence._channels):
            raise ValueError("No instructions given for the channels in the "
                             "sequence.")
        self._seq = sequence
        self._qdict = self._seq.qubit_info
        self._size = len(self._qdict)
        self._tot_duration = max(
            [self._seq._last(ch).tf for ch in self._seq._schedule]
        )
        if not (0 < sampling_rate <= 1.0):
            raise ValueError("`sampling_rate` must be positive and "
                             "not larger than 1.0")
        if int(self._tot_duration*sampling_rate) < 4:
            raise ValueError("`sampling_rate` is too small, less than 4 data "
                             "points.")
        self.sampling_rate = sampling_rate
        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}
        self.samples = {addr: {basis: {}
                               for basis in ['ground-rydberg', 'digital']}
                        for addr in ['Global', 'Local']}
        self.operators = deepcopy(self.samples)
        self._noise = []
        self._collapse_ops = []
        self._temperature = 50e-6
        self._build_hamiltonian()
        self._init_config()
        self.config('eval_t', evaluation_times)

    def _set_eval_times(self, evaluation_times):
        if isinstance(evaluation_times, str):
            if evaluation_times == 'Full':
                self.eval_times = deepcopy(self._times)
            elif evaluation_times == 'Minimal':
                self.eval_times = np.array([self._times[0], self._times[-1]])
            else:
                raise ValueError("Wrong evaluation time label. It should "
                                 "be `Full` or `Minimal`")
        elif isinstance(evaluation_times, (list, tuple, np.ndarray)):
            t_max = np.max(evaluation_times)
            t_min = np.min(evaluation_times)
            if t_max > self._times[-1]:
                raise ValueError("Provided evaluation-time list extends "
                                 "further than sequence duration.")
            if t_min < 0:
                raise ValueError("Provided evaluation-time list contains "
                                 "negative values.")
            # Ensure the list of times is sorted
            eval_times = np.array(np.sort(evaluation_times))
            if t_min > 0:
                eval_times = np.insert(eval_times, 0, 0.)
            if t_max < self._times[-1]:
                eval_times = np.append(eval_times, self._times[-1])
            self.eval_times = eval_times
            # always include initial and final times
        else:
            raise ValueError("`evaluation_times` must be a list of times "
                             "or `Full` or `Minimal`")

    def draw(self):
        """Draws the input sequence and the one used in QuTip."""

        draw_sequence(self._seq, self.sampling_rate)

    def _extract_samples(self):
        """Populate samples dictionary with every pulse in the sequence."""
        def prepare_dict():
            # Duration includes retargeting, delays, etc.
            return {'amp': np.zeros(self._tot_duration),
                    'det': np.zeros(self._tot_duration),
                    'phase': np.zeros(self._tot_duration)}

        def write_samples(slot, samples_dict, qid=None):
            """constructs hamiltonian coefficients, taking into account, if
                necessary, noise errors, which are local and depend
                on the qubit's id qid"""
            noise_det = 0
            noise_amp = 1
            # detuning offset related to bad preparation
            # not too large, or else ODE integration errors
            det_spam = 150
            for noise in self._noise:
                if noise == 'doppler':
                    noise_det += np.random.normal(0, self._doppler_sigma)
                # qubit qid badly prepared
                if noise == 'SPAM' and self.spam_detune[qid]:
                    noise_det += det_spam
                # gaussian beam for global pulses
                if noise == 'amplitude':
                    position = self._qdict[qid]
                    r = np.linalg.norm(position)
                    w0 = 175.
                    noise_amp = np.random.normal(1, 1e-3) * np.exp(-(r/w0)**2)

            samples_dict['amp'][slot.ti:slot.tf] = \
                slot.type.amplitude.samples * noise_amp
            samples_dict['det'][slot.ti:slot.tf] = \
                slot.type.detuning.samples + noise_det
            samples_dict['phase'][slot.ti:slot.tf] = slot.type.phase

        for channel in self._seq.declared_channels:
            addr = self._seq.declared_channels[channel].addressing
            basis = self._seq.declared_channels[channel].basis

            # Case of clean global simulations
            if addr == 'Global' and not self._noise:
                samples_dict = self.samples[addr][basis]
                if not samples_dict:
                    samples_dict = prepare_dict()
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        write_samples(slot, samples_dict)
                self.samples[addr][basis] = samples_dict

            # any noise : global becomes local for each qubit in the reg
            elif addr == 'Global':
                samples_dict = self.samples['Local'][basis]
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        # global to local
                        for qubit in self._qid_index:
                            if qubit not in samples_dict:
                                samples_dict[qubit] = prepare_dict()
                            write_samples(slot, samples_dict[qubit], qid=qubit)
                self.samples['Local'][basis] = samples_dict

            elif addr == 'Local':
                samples_dict = self.samples['Local'][basis]
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        for qubit in slot.targets:  # Allow multiaddressing
                            if qubit not in samples_dict:
                                samples_dict[qubit] = prepare_dict()
                            write_samples(slot, samples_dict[qubit], qid=qubit)
                self.samples[addr][basis] = samples_dict

    def _build_basis_and_op_matrices(self):
        """Determine dimension, basis and projector operators."""
        # No samples => Empty dict entry => False
        if (not self.samples['Global']['digital']
                and not self.samples['Local']['digital']):
            self.basis_name = 'ground-rydberg'
            self.dim = 2
            basis = ['r', 'g']
            projectors = ['gr', 'rr', 'gg']
        elif (not self.samples['Global']['ground-rydberg']
                and not self.samples['Local']['ground-rydberg']):
            self.basis_name = 'digital'
            self.dim = 2
            basis = ['g', 'h']
            projectors = ['hg', 'hh', 'gg']
        else:
            self.basis_name = 'all'  # All three states
            self.dim = 3
            basis = ['r', 'g', 'h']
            projectors = ['gr', 'hg', 'rr', 'gg', 'hh', 'hr']

        self.basis = {b: qutip.basis(self.dim, i) for i, b in enumerate(basis)}
        self.op_matrix = {'I': qutip.qeye(self.dim)}

        for proj in projectors:
            self.op_matrix['sigma_' + proj] = (
                self.basis[proj[0]] * self.basis[proj[1]].dag()
            )

    def _build_operator(self, op_id, *qubit_ids, global_op=False):
        """Create qutip.Qobj with nontrivial action at *qubit_ids."""
        if global_op:
            return sum(self._build_operator(op_id, q_id)
                       for q_id in self._qdict)
        if len(set(qubit_ids)) < len(qubit_ids):
            raise ValueError("Duplicate atom ids in argument list.")
        # List of identity operators, except for op_id where requested:
        op_list = [self.op_matrix[op_id]
                   if j in map(self._qid_index.get, qubit_ids)
                   else self.op_matrix['I'] for j in range(self._size)]
        return qutip.tensor(op_list)

    def _construct_hamiltonian(self):
        def adapt(full_array):
            """Adapt list to correspond to sampling rate"""
            indexes = np.linspace(0, self._tot_duration-1,
                                  int(self.sampling_rate*self._tot_duration),
                                  dtype=int)
            return full_array[indexes]

        def make_vdw_term():
            """Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate the distance between them, then
            assign the local operator "sigma_rr" at each pair. The units are
            given so that the coefficient includes a 1/hbar factor.
            """
            vdw = 0
            # Get every pair without duplicates
            for q1, q2 in itertools.combinations(self._qdict.keys(), r=2):
                dist = np.linalg.norm(
                    self._qdict[q1] - self._qdict[q2])
                U = 0.5 * self._seq._device.interaction_coeff / dist**6
                vdw += U * self._build_operator('sigma_rr', q1, q2)
            return vdw

        def build_coeffs_ops(basis, addr):
            """Build coefficients and operators for the hamiltonian QobjEvo."""
            samples = self.samples[addr][basis]
            operators = self.operators[addr][basis]
            # Choose operator names according to addressing:
            if basis == 'ground-rydberg':
                op_ids = ['sigma_gr', 'sigma_rr']
            elif basis == 'digital':
                op_ids = ['sigma_hg', 'sigma_gg']

            terms = []
            if addr == 'Global':
                coeffs = [0.5*samples['amp'] * np.exp(-1j * samples['phase']),
                          -0.5 * samples['det']]
                for op_id, coeff in zip(op_ids, coeffs):
                    if np.any(coeff != 0):
                        # Build once global operators as they are needed
                        if op_id not in operators:
                            operators[op_id] =\
                                self._build_operator(op_id, global_op=True)
                        terms.append([operators[op_id], adapt(coeff)])
            elif addr == 'Local':
                for q_id, samples_q in samples.items():
                    if q_id not in operators:
                        operators[q_id] = {}
                    coeffs = [0.5*samples_q['amp'] *
                              np.exp(-1j * samples_q['phase']),
                              -0.5 * samples_q['det']]
                    for coeff, op_id in zip(coeffs, op_ids):
                        if np.any(coeff != 0):
                            if op_id not in operators[q_id]:
                                operators[q_id][op_id] = \
                                    self._build_operator(op_id, q_id)
                            terms.append([operators[q_id][op_id],
                                          adapt(coeff)])

            self.operators[addr][basis] = operators
            return terms

        # Time independent term:
        if self.basis_name == 'digital':
            qobj_list = []
        else:
            # Van der Waals Interaction Terms
            qobj_list = [make_vdw_term()] if self._size > 1 else []

        # Time dependent terms:
        for addr in self.samples:
            for basis in self.samples[addr]:
                if self.samples[addr][basis]:
                    qobj_list += build_coeffs_ops(basis, addr)

        self._times = adapt(np.arange(self._tot_duration,
                                      dtype=np.double)/1000)

        ham = qutip.QobjEvo(qobj_list, tlist=self._times)
        ham = ham + ham.dag()
        ham.compress()

        self._hamiltonian = ham

    def _init_config(self):
        """Sets default configuration for the Simulation."""
        all_ground = qutip.tensor([self.basis['g'] for _ in range(self._size)])
        self._set_eval_times('Full')
        self._config = {'eval_t': 'Full', 'runs': 1, 'samples_per_run': 10,
                        'initial_state': all_ground}

    def get_hamiltonian(self, time):
        """Get the Hamiltonian created from the sequence at a fixed time.

        Args:
            time (float): The specific time in which we want to extract the
                    Hamiltonian (in ns).

        Returns:
            qutip.Qobj: A new Qobj for the Hamiltonian with coefficients
            extracted from the effective sequence (determined by
            `self.sampling_rate`) at the specified time.
        """
        # Refresh the hamiltonian
        self._build_hamiltonian()
        if time > 1000 * self._times[-1]:
            raise ValueError("Provided time is larger than sequence duration.")
        if time < 0:
            raise ValueError("Provided time is negative.")
        return self._hamiltonian(time/1000)  # Creates new Qutip.Qobj

    def _prepare_spam_detune(self):
        """Choose atoms that will be badly prepared"""
        # Tag True if atom is badly prepared
        self.spam_detune = {qid: (np.random.uniform() < self._spam_dict['eta'])
                            for qid in self._qid_index}

    @property
    def doppler_sigma(self):
        return self._doppler_sigma

    @property
    def temperature(self):
        return self._temperature

    @property
    def spam_dict(self):
        return self._spam_dict

    @temperature.setter
    def temperature(self, value):
        # value set in microkelvin
        self._temperature = value * 10**-6
        self._calc_sigma_doppler()

    def _calc_sigma_doppler(self):
        # sigma = keff Deltav, keff = 8.7mum^-1, Deltav = sqrt(kB T / m)
        self._doppler_sigma = 8.7 * np.sqrt(1.38e-23 * self._temperature /
                                            1.45e-25)

    @doppler_sigma.setter
    def doppler_sigma(self, value):
        self._doppler_sigma = value

    def init_doppler_sigma(self):
        self._doppler_sigma = 8.7 * np.sqrt(1.38e-23 * 50e-6 /
                                            1.45e-25)

    def init_spam(self):
        self._spam_dict = {'eta': 0.005, 'epsilon': 0.01,
                           'epsilon_prime': 0.05}

    def add_noise(self, noise_type):
        """Adds a noise model to the Simulation instance.
            Args:
                noise_type (str): Choose among:
                    'doppler': Noisy doppler runs
                    'amplitude: Noisy gaussian beam
                    'SPAM': SPAM errors. Adds:
                        eta: Probability of each atom to be badly prepared
                        epsilon: false positives
                        epsilon_prime: false negatives
        """
        # Check proper input:
        noise_dict_set = {'doppler', 'amplitude', 'SPAM', 'dephasing'}
        if noise_type not in noise_dict_set:
            raise ValueError('Not a valid noise type')
        if noise_type in set(self._noise):
            print("Noise type already in place. Skipping instruction.")
            return

        # Add noise/error:
        if noise_type == 'SPAM':
            # Set SPAM parameters (experimental)
            self.init_spam()
            self._prepare_spam_detune()
        if noise_type == 'doppler':
            self.init_doppler_sigma()
        if noise_type == 'dephasing':
            if self.basis_name == 'digital' or self.basis_name == 'all':
                raise ValueError("Cannot include dephasing noise in digital-"
                                 "or all-basis.")
            self.init_dephasing()
        # Register added noise(s)
        self._noise.append(noise_type)
        # Reset any previous sequence:
        self.reset_sequence()

    def remove_all_noise(self):
        """Removes noise from simulation"""
        self._noise = []
        # Reset any previously sequence:
        self.reset_sequence()

    def reset_sequence(self):
        self.samples = {addr: {basis: {}
                               for basis in ['ground-rydberg', 'digital']}
                        for addr in ['Global', 'Local']}

    def set_spam(self, **values):
        """Allows the user to change SPAM parameters in dictionary"""
        for param in values:
            if param not in {'eta', 'epsilon', 'epsilon_prime'}:
                raise ValueError('Not a valid SPAM parameter')
            self._spam_dict[param] = values[param]

    def init_dephasing(self):
        self.dephasing_prob = 0.05  # Probability of phase (Z) flip
        self._collapse_ops += [
            np.sqrt(1 - self.dephasing_prob) * self.op_matrix['I'],
            np.sqrt(self.dephasing_prob) * (self.op_matrix['sigma_rr']
                                            - self.op_matrix['sigma_gg'])]

    def set_noise(self, *noise_param):
        """Sets noise parameters as those in argument"""
        self.remove_all_noise()
        for noise in noise_param:
            self.add_noise(noise)

    def config(self, parameter, value):
        """Include additional parameters to simulation. Will be necessary for
        noisy simulations.

        Allowed 'parameter' strings:
            'initial_state (array)': The initial quantum state of the
                evolution. Will be transformed into a
                qutip.Qobj instance.
            'eval_t' (str / array): Time at which the results are to be
                returned. Allowed values : 'Full', 'Minimal', or an array of
                times.
            'samples_per_run' (int): number of samples per noisy run. Useful
                for cutting down on computing time, but unrealistic.
            'runs' (int): number of runs needed : each run
                draws a new random noise.
        """

        if parameter not in self._config:
            raise ValueError('Not a valid setting')

        else:
            if parameter == 'initial_state':
                if isinstance(value, qutip.Qobj):
                    if value.shape != (self.dim**self._size, 1):
                        raise ValueError("Incompatible shape of initial_state")
                    self._config[parameter] = value
                else:
                    if value.shape != (self.dim**self._size,):
                        raise ValueError("Incompatible shape of initial_state")
                    self._config[parameter] = qutip.Qobj(value)
            else:
                self._config[parameter] = value
                if parameter == 'eval_t':
                    self._set_eval_times(value)

    def show_config(self):
        print(self._config)

    def reset_config(self):
        self._init_config()
        print('Configuration has been set to default')

    def _build_hamiltonian(self):
        """Extracts the sequence samples, builds default operators and
        builds the hamiltonian from those coefficients and operators."""
        self._extract_samples()
        self._build_basis_and_op_matrices()
        self._construct_hamiltonian()

    # Run Simulation Evolution using Qutip
    def run(self, progress_bar=None, **options):
        """Simulate the sequence using QuTiP's solvers. Only clean results
            are returned.

        Keyword Args:
            progress_bar (bool): If True, the progress bar of QuTiP's sesolve()
                will be shown.
        """
        def _assign_meas_basis():
            if hasattr(self._seq, '_measurement'):
                return self._seq._measurement
            else:
                if self.basis_name in {'digital', 'all'}:
                    return 'digital'
                else:
                    return 'ground-rydberg'

        def _run_solver(as_subroutine=False,
                        measurement_basis=None):
            """Returns:
                CleanResults: Object containing the time evolution results. Its
                _states attribute contains QuTiP quantum states, not Counters.
            """
            if not as_subroutine:
                # CLEAN SIMULATION:
                # Build (clean) hamiltonian
                self._build_hamiltonian()
                measurement_basis = _assign_meas_basis()

            result = qutip.mesolve(self._hamiltonian,
                                   self._config['initial_state'],
                                   self.eval_times,
                                   c_ops=self._collapse_ops,
                                   progress_bar=progress_bar,
                                   options=qutip.Options(max_step=5,
                                                         **options)
                                   )
            return CleanResults(result.states, self.dim, self._size,
                                self.basis_name, measurement_basis,
                                self.eval_times)

        if self._noise:
            # NOISY SIMULATION:
            # We run the system multiple times
            time_indices = range(len(self.eval_times))
            total_count = np.array([Counter() for _ in time_indices])
            for _ in range(self._config['runs']):
                # At each run, new random noise
                if 'SPAM' in self._noise:
                    self._prepare_spam_detune()
                self._build_hamiltonian()
                meas_basis = _assign_meas_basis()
                # Get CleanResults instance from sequence with added noise:
                clean_res_noisy_seq = _run_solver(as_subroutine=True,
                                                  measurement_basis=meas_basis)
                # Extract statistics at eval time (final time by default):
                if 'SPAM' in self._noise:
                    total_count += \
                        [clean_res_noisy_seq.sampling_with_detection_errors(
                            self._spam_dict,
                            t=t,
                            meas_basis=meas_basis,
                            N_samples=self._config['samples_per_run'])
                         for t in time_indices]
                else:
                    total_count += \
                        [clean_res_noisy_seq.sample_state(
                            t=t,
                            meas_basis=meas_basis,
                            N_samples=self._config['samples_per_run'])
                         for t in time_indices]
            N_measures = self._config['runs'] * self._config['samples_per_run']
            total_run_prob = [Counter({k: v / N_measures
                                      for k, v in total_count[t].items()})
                              for t in time_indices]
            return NoisyResults(total_run_prob, self._size, self.basis_name,
                                meas_basis, self.eval_times, N_measures)
        else:
            return _run_solver()
