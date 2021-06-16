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
"""Contains the Simulation class, used for simulation of a Sequence."""

from __future__ import annotations

from typing import Optional, Union, cast, List
from collections.abc import Mapping
import itertools
from collections import Counter
from copy import deepcopy

import qutip
import numpy as np
from numpy.typing import ArrayLike

from pulser import Pulse, Sequence
from pulser.simulation.simresults import SimulationResults, \
                                         CleanResults, NoisyResults
from pulser.simulation.simconfig import SimConfig
from pulser._seq_drawer import draw_sequence
from pulser.sequence import _TimeSlot


class Simulation:
    """Simulation of a pulse sequence using QuTiP.

    Provides methods to simulate the sequence using QuTiP.

    Args:
        sequence (Sequence): An instance of a Pulser Sequence that we
            want to simulate.
        sampling_rate (float): The fraction of samples that we wish to
            extract from the pulse sequence to simulate. Has to be a
            value between 0.05 and 1.0.
        config (SimConfig): Configuration to be used for this simulation.
        evaluation_times (Union[str, list, float]: The list of times at
            which the quantum state should be evaluated, in μs.
            If 'Full' is provided, this list is set to be the one used to
            define the Hamiltonian to the solver.
            The initial and final times are always included, so that if
            'Minimal' is provided, the list is set to only contain the
            initial and the final times. Use a float to act as a sampling
            rate for the resulting state.
    """

    def __init__(self, sequence: Sequence, sampling_rate: float = 1.0,
                 config: Optional[SimConfig] = None,
                 evaluation_times: Union[float, str, ArrayLike] = 'Full'
                 ) -> None:
        """Instantiates a Simulation object."""
        self.seq = sequence
        self._qdict = self._seq.qubit_info
        self._size = len(self._qdict)
        self._tot_duration = max(
            [self._seq._last(ch).tf for ch in self._seq._schedule]
        )
        self.sampling_rate = sampling_rate
        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}
        self.samples: dict[str, dict[str, dict]] = {
            addr: {basis: {} for basis in ['ground-rydberg', 'digital']}
            for addr in ['Global', 'Local']}
        self.operators = deepcopy(self.samples)
        self._times = self._adapt_to_sampling_rate(
            np.arange(self._tot_duration, dtype=np.double)/1000)
        self.evaluation_times = evaluation_times
        self.config = config if config else SimConfig()
        self.initial_state = 'all-ground'
        self._collapse_ops: List[qutip.Qobj] = []

    @property
    def sampling_rate(self) -> float:
        """Property getter for sampling_rate."""
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        if not (0 < value <= 1.0):
            raise ValueError("`sampling_rate` must be positive and "
                             "not larger than 1.0")
        if int(self._tot_duration * value) < 4:
            raise ValueError("`sampling_rate` is too small, less than 4 data "
                             "points.")
        self._sampling_rate: float = value

    @property
    def seq(self) -> Sequence:
        """Property getter for sequence."""
        return self._seq

    @seq.setter
    def seq(self, seq: Sequence) -> None:
        if not isinstance(seq, Sequence):
            raise TypeError("The provided sequence has to be a valid "
                            "pulser.Sequence instance.")
        if not seq._schedule:
            raise ValueError("The provided sequence has no declared channels.")
        if all(seq._schedule[x][-1].tf == 0 for x in seq._channels):
            raise ValueError("No instructions given for the channels in the "
                             "sequence.")
        self._seq = seq

    @property
    def config(self) -> SimConfig:
        """Property getter for config."""
        return self._config

    @config.setter
    def config(self, cfg: SimConfig) -> None:
        self._config = cfg
        self._set_param_from_config()

    def show_config(self) -> None:
        """Shows current configuration."""
        print(self._config)

    def reset_config(self) -> None:
        """Resets configuration to default."""
        self.config = SimConfig()
        print("Configuration has been set to default.")

    def _set_param_from_config(self) -> None:
        """Sets all relevant Simulation parameters from its SimConfig.

        Called every time a new configuration is loaded in order to update
        Simulation parameters.
        """
        self._prepare_bad_atoms()
        # update samples with potential noise settings
        self._build_hamiltonian_from_seq()
        if 'dephasing' in self.config.noise:
            self._init_dephasing()

    def _prepare_bad_atoms(self) -> None:
        """Chooses atoms that will be badly prepared.

        If no SPAM noise, simply sets all atoms as well-prepared.
        """
        # Tag True if atom is badly prepared
        self._bad_atoms: dict[Union[str, int], bool] = {
            qid: False for qid in self._qid_index}
        if 'SPAM' in self.config.noise:
            self._bad_atoms = \
                {qid: (np.random.uniform() < self.config.spam_dict['eta'])
                 for qid in self._qid_index}

    @property
    def initial_state(self) -> qutip.Qobj:
        """Property getter for initial_state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value: Union[str, ArrayLike, qutip.qObj]) -> None:
        self._initial_state: qutip.Qobj
        if value == 'all-ground':
            self._initial_state = \
                qutip.tensor([self.basis['g'] for _ in range(self._size)])
        else:
            if isinstance(value, qutip.Qobj):
                if cast(qutip.Qobj, value).shape != \
                        (self.dim**self._size, 1):
                    raise ValueError("Incompatible shape of initial_state")
                self._initial_state = value
            else:
                if cast(np.ndarray, value).shape != \
                        (self.dim**self._size,):
                    raise ValueError("Incompatible shape of initial_state")
                self._initial_state = qutip.Qobj(value)

    @property
    def evaluation_times(self) -> Union[str, float, ArrayLike]:
        """Property getter for evaluation_times."""
        return self._evaluation_times

    @evaluation_times.setter
    def evaluation_times(self, value: Union[str, ArrayLike, float]) -> None:
        if isinstance(value, str):
            if value == 'Full':
                self._eval_times_array = self._times
            elif value == 'Minimal':
                self._eval_times_array = np.array([self._times[0],
                                                  self._times[-1]])
            else:
                raise ValueError("Wrong evaluation time label. It should "
                      "be `Full` or `Minimal` or a float between 0 and 1.")
        elif isinstance(value, float):
            if value > 1 or value <= 0:
                raise ValueError("evaluation_times float must be between 0 "
                                 "and 1.")
            indices = np.linspace(0, len(self._times)-1,
                                  int(value * len(self._times)),
                                  dtype=int)
            self._eval_times_array = self._times[indices]
        elif isinstance(value, (list, tuple, np.ndarray)):
            t_max = np.max(value)
            t_min = np.min(value)
            if t_max > self._times[-1]:
                raise ValueError("Provided evaluation-time list extends "
                                 "further than sequence duration.")
            if t_min < 0:
                raise ValueError("Provided evaluation-time list contains "
                                 "negative values.")
            # Ensure the list of times is sorted
            eval_times = np.array(np.sort(value))
            if t_min > 0:
                eval_times = np.insert(eval_times, 0, 0.)
            if t_max < self._times[-1]:
                eval_times = np.append(eval_times, self._times[-1])
            self._eval_times_array = eval_times
            # always include initial and final times
        else:
            raise ValueError("Wrong evaluation time label. It should "
                  "be `Full` or `Minimal` or a float between 0 and 1.")
        self._evaluation_times: Union[str, ArrayLike, float] = value

    def draw(self, draw_phase_area: bool = False) -> None:
        """Draws the input sequence and the one used in QuTip.

        Keyword Args:
            draw_phase_area (bool): Whether phase and area values need
                to be shown as text on the plot, defaults to False.
        """
        draw_sequence(
            self._seq, self.sampling_rate, draw_phase_area=draw_phase_area
        )

    def _extract_samples(self) -> None:
        """Populates samples dictionary with every pulse in the sequence."""

        def prepare_dict() -> dict[str, np.ndarray]:
            # Duration includes retargeting, delays, etc.
            return {'amp': np.zeros(self._tot_duration),
                    'det': np.zeros(self._tot_duration),
                    'phase': np.zeros(self._tot_duration)}

        def write_samples(slot: _TimeSlot,
                          samples_dict: Mapping[str, np.ndarray],
                          *qid: Union[int, str]) -> None:
            """Builds hamiltonian coefficients.

            Taking into account, if necessary, noise effects, which are local
            and depend on the qubit's id qid.
            """
            _pulse = cast(Pulse, slot.type)
            noise_det = 0.
            noise_amp = 1.
            for noise in self.config.noise:
                if noise == 'doppler':
                    noise_det += np.random.normal(
                        0, self.config.doppler_sigma)
                # Gaussian beam for global pulses
                if noise == 'amplitude':
                    position = self._qdict[qid[0]]
                    r = np.linalg.norm(position)
                    w0 = self.config.laser_waist
                    noise_amp = np.random.normal(1., 1.e-3) * \
                        np.exp(-(r/w0)**2)
            samples_dict['amp'][slot.ti:slot.tf] = \
                _pulse.amplitude.samples * noise_amp
            samples_dict['det'][slot.ti:slot.tf] = \
                _pulse.detuning.samples + noise_det
            samples_dict['phase'][slot.ti:slot.tf] = _pulse.phase

        for channel in self._seq.declared_channels:
            addr = self._seq.declared_channels[channel].addressing
            basis = self._seq.declared_channels[channel].basis

            # Case of clean global simulations
            if addr == 'Global' and not self.config.noise:
                samples_dict = self.samples[addr][basis]
                if not samples_dict:
                    samples_dict = prepare_dict()
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        write_samples(slot, samples_dict)
                self.samples[addr][basis] = samples_dict

            # Any noise : global becomes local for each qubit in the reg
            # Since coefficients are modified locally by all noises
            elif addr == 'Global':
                samples_dict = self.samples['Local'][basis]
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        # global to local
                        for qubit in self._qid_index:
                            if qubit not in samples_dict:
                                samples_dict[qubit] = prepare_dict()
                            # We don't write samples for badly prep qubits
                            if not self._bad_atoms[qubit]:
                                write_samples(slot, samples_dict[qubit], qubit)
                self.samples['Local'][basis] = samples_dict

            # Local noisy pulse
            elif addr == 'Local':
                samples_dict = self.samples['Local'][basis]
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        for qubit in slot.targets:  # Allow multiaddressing
                            if qubit not in samples_dict:
                                samples_dict[qubit] = prepare_dict()
                            if not self._bad_atoms[qubit]:
                                write_samples(slot, samples_dict[qubit], qubit)
                self.samples[addr][basis] = samples_dict

    def _build_basis_and_op_matrices(self) -> None:
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

    def _build_operator(self, op_id: str, *qubit_ids: Union[str, int],
                        global_op: bool = False) -> qutip.Qobj:
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

    def _adapt_to_sampling_rate(self, full_array: np.ndarray) -> np.ndarray:
        """Adapt list to correspond to sampling rate."""
        indices = np.linspace(0, self._tot_duration-1,
                              int(self.sampling_rate*self._tot_duration),
                              dtype=int)
        return cast(np.ndarray, full_array[indices])

    def _construct_hamiltonian(self) -> None:
        def make_vdw_term() -> qutip.Qobj:
            """Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate the distance between them, then
            assign the local operator "sigma_rr" at each pair. The units are
            given so that the coefficient includes a 1/hbar factor.
            """
            vdw = 0 * self._build_operator('I')
            # Get every pair without duplicates
            for q1, q2 in itertools.combinations(self._qdict.keys(), r=2):
                # no VdW interaction with other qubits for a badly prep. qubit
                if not(self._bad_atoms[q1] or self._bad_atoms[q2]):
                    dist = np.linalg.norm(self._qdict[q1] - self._qdict[q2])
                    U = 0.5 * self._seq._device.interaction_coeff / dist**6
                    vdw += U * self._build_operator('sigma_rr', q1, q2)
            return vdw

        def build_coeffs_ops(basis: str, addr: str) -> list[list]:
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
                        terms.append([operators[op_id],
                                     self._adapt_to_sampling_rate(coeff)])
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
                                          self._adapt_to_sampling_rate(coeff)])

            self.operators[addr][basis] = operators
            return terms

        # Time independent term:
        if self.basis_name == 'digital' or self._size == 1:
            qobj_list = []
        else:
            # Van der Waals Interaction Terms
            qobj_list = [make_vdw_term()]

        # Time dependent terms:
        for addr in self.samples:
            for basis in self.samples[addr]:
                if self.samples[addr][basis]:
                    qobj_list += cast(list, build_coeffs_ops(basis, addr))
        if not qobj_list:
            # can't simulate an empty sequence
            qobj_list = [0*self._build_operator('I')]
        ham = qutip.QobjEvo(qobj_list, tlist=self._times)
        ham = ham + ham.dag()
        ham.compress()

        self._hamiltonian = ham

    def get_hamiltonian(self, time: float) -> qutip.Qobj:
        """Get the Hamiltonian created from the sequence at a fixed time.

        Args:
            time (float): The specific time in which we want to extract the
                    Hamiltonian (in ns).

        Returns:
            qutip.Qobj: A new Qobj for the Hamiltonian with coefficients
            extracted from the effective sequence (determined by
            `self.sampling_rate`) at the specified time.
        """
        if time > 1000 * self._times[-1]:
            raise ValueError(f"Provided time (`time` = {time}) must be "
                             "less than or equal to the sequence duration "
                             f"({1000 * self._times[-1]}).")
        if time < 0:
            raise ValueError(f"Provided time (`time` = {time}) must be "
                             "greater than or equal to 0.")
        return self._hamiltonian(time/1000)  # Creates new Qutip.Qobj

    def _reset_samples(self) -> None:
        """Sets all samples to their default values.

        Used when running several noisy runs, each one setting new samples.
        """
        self.samples = {addr: {basis: {}
                               for basis in ['ground-rydberg', 'digital']}
                        for addr in ['Global', 'Local']}

    def _init_dephasing(self) -> None:
        """Initializes dephasing collapse operators."""
        if self.basis_name == 'digital' or self.basis_name == 'all':
            raise ValueError("Cannot include dephasing noise in digital-"
                             "or all-basis.")
        self.dephasing_prob = 0.05  # Probability of phase (Z) flip
        self._collapse_ops += [
            np.sqrt(1 - self.dephasing_prob) * self.op_matrix['I'],
            np.sqrt(self.dephasing_prob) * (self.op_matrix['sigma_rr']
                                            - self.op_matrix['sigma_gg'])]

    def _build_hamiltonian_from_seq(self) -> None:
        """Builds the hamiltonian from sequence samples.

        Updates the new sequence samples that may have changed due to noise,
        builds default operators and builds the hamiltonian from those
        coefficients and operators.
        """
        self._reset_samples()
        self._extract_samples()
        self._build_basis_and_op_matrices()
        self._construct_hamiltonian()

    # Run Simulation Evolution using Qutip
    def run(self, progress_bar: Optional[bool] = None,
            **options: qutip.solver.Options) -> SimulationResults:
        """Simulate the sequence using QuTiP's solvers.

        Will return NoisyResults if it detects any noise in the SimConfig.
        Otherwise will return CleanResults.

        Keyword Args:
            progress_bar (bool): If True, the progress bar of QuTiP's solver
                will be shown.
            options (qutip.solver.Options): If specified, will override
                SimConfig solver_options.
        """
        solv_ops = qutip.Options(max_step=5, **options) if options \
            else self.config.solver_options

        def _assign_meas_basis() -> str:
            if hasattr(self._seq, '_measurement'):
                return cast(str, self._seq._measurement)
            else:
                if self.basis_name in {'digital', 'all'}:
                    return 'digital'
                else:
                    return 'ground-rydberg'

        def _run_solver(as_subroutine: bool = False,
                        measurement_basis: str = '') -> CleanResults:
            """Returns CleanResults: Object containing evolution results.

            Its _states attribute contains QuTiP quantum states, not Counters.
            """
            if not as_subroutine:
                # CLEAN SIMULATION:
                measurement_basis = _assign_meas_basis()
            if self._collapse_ops:
                # temporary workaround due to a qutip bug when using mesolve
                liouvillian = qutip.liouvillian(self._hamiltonian,
                                                self._collapse_ops)
                result = qutip.mesolve(liouvillian,
                                       self.initial_state,
                                       self._eval_times_array,
                                       progress_bar=progress_bar,
                                       options=solv_ops)
            else:
                result = qutip.sesolve(self._hamiltonian,
                                       self.initial_state,
                                       self._eval_times_array,
                                       progress_bar=progress_bar,
                                       options=solv_ops)
            return CleanResults(result.states, self._size,
                                self.basis_name, self._eval_times_array,
                                measurement_basis)

        if self.config.noise:
            # NOISY SIMULATION:
            meas_basis = _assign_meas_basis()
            if self.basis_name == 'all':
                basis_name = 'digital'
            else:
                basis_name = self.basis_name

            time_indices = range(len(self._eval_times_array))
            total_count = np.array([Counter() for _ in time_indices])
            # We run the system multiple times
            for _ in range(self.config.runs):
                # At each run, new random noise: new Hamiltonian
                if 'SPAM' in self.config.noise:
                    self._prepare_bad_atoms()
                self._build_hamiltonian_from_seq()
                # Get CleanResults instance from sequence with added noise:
                clean_res_noisy_seq = _run_solver(as_subroutine=True,
                                                  measurement_basis=meas_basis)
                # Extract statistics at eval time:
                if 'SPAM' in self.config.noise:
                    total_count += np.array(
                        [clean_res_noisy_seq._sampling_with_detection_errors(
                            self.config.spam_dict,
                            t, n_samples=self.config.samples_per_run)
                         for t in self._eval_times_array])
                else:
                    total_count += np.array(
                        [clean_res_noisy_seq.sample_state(
                            t, n_samples=self.config.samples_per_run)
                         for t in self._eval_times_array])
            n_measures = self.config.runs * self.config.samples_per_run
            total_run_prob = [Counter({k: v / n_measures
                                      for k, v in total_count[t].items()})
                              for t in time_indices]
            return NoisyResults(total_run_prob, self._size, basis_name,
                                self._eval_times_array, n_measures)
        else:
            return _run_solver()
