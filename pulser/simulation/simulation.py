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

from typing import Optional, Union, cast
from collections.abc import Mapping
import itertools

import qutip
import numpy as np
from numpy.typing import ArrayLike
from copy import deepcopy

from pulser import Pulse, Sequence
from pulser.simulation.simresults import SimulationResults
from pulser._seq_drawer import draw_sequence
from pulser.sequence import _TimeSlot


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

    def __init__(self, sequence: Sequence, sampling_rate: float = 1.0,
                 evaluation_times: Union[str, ArrayLike] = 'Full') -> None:
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
        self._interaction = 'XY' if self._seq._in_xy else 'ising'

        if not (0 < sampling_rate <= 1.0):
            raise ValueError("`sampling_rate` must be positive and "
                             "not larger than 1.0")
        if int(self._tot_duration*sampling_rate) < 4:
            raise ValueError("`sampling_rate` is too small, less than 4 data "
                             "points.")
        self.sampling_rate = sampling_rate

        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}

        if self._interaction == 'ising':
            self.samples = {addr: {basis: {}
                            for basis in ['ground-rydberg', 'digital']}
                            for addr in ['Global', 'Local']}
        else:
            self.samples = {addr: {'XY': {}}
                            for addr in ['Global', 'Local']}
        self.operators = deepcopy(self.samples)

        self._extract_samples()
        self._build_basis_and_op_matrices()
        self._construct_hamiltonian()

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

    def draw(self, draw_phase_area: bool = False) -> None:
        """Draws the input sequence and the one used in QuTip.

        Keyword args:
            draw_phase_area (bool): Whether phase and area values need
                to be shown as text on the plot, defaults to False.
        """
        draw_sequence(
            self._seq, self.sampling_rate, draw_phase_area=draw_phase_area
        )

    def _extract_samples(self) -> None:
        """Populate samples dictionary with every pulse in the sequence."""

        def prepare_dict() -> dict[str, np.ndarray]:
            # Duration includes retargeting, delays, etc.
            return {'amp': np.zeros(self._tot_duration),
                    'det': np.zeros(self._tot_duration),
                    'phase': np.zeros(self._tot_duration)}

        def write_samples(slot: _TimeSlot,
                          samples_dict: Mapping[str, np.ndarray]) -> None:
            _pulse = cast(Pulse, slot.type)
            samples_dict['amp'][slot.ti:slot.tf] += _pulse.amplitude.samples
            samples_dict['det'][slot.ti:slot.tf] += _pulse.detuning.samples
            samples_dict['phase'][slot.ti:slot.tf] = _pulse.phase

        for channel in self._seq.declared_channels:
            addr = self._seq.declared_channels[channel].addressing
            basis = self._seq.declared_channels[channel].basis

            samples_dict = self.samples[addr][basis]

            if addr == 'Global':
                if not samples_dict:
                    samples_dict = prepare_dict()
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        write_samples(slot, samples_dict)

            elif addr == 'Local':
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        for qubit in slot.targets:  # Allow multiaddressing
                            if qubit not in samples_dict:
                                samples_dict[qubit] = prepare_dict()
                            write_samples(slot, samples_dict[qubit])

            self.samples[addr][basis] = samples_dict

    def _build_basis_and_op_matrices(self) -> None:
        """Determine dimension, basis and projector operators."""
        if self._interaction == 'XY':
            self.basis_name = 'XY'
            self.dim = 2
            basis = ['u', 'd']
            projectors = ['uu', 'du', 'ud', 'dd']
        else:
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
                projectors = ['gr', 'hg', 'rr', 'gg', 'hh']

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

    def _build_general_operator(
        self, op_id: list[str], qubit_ids: list[Union[str, int]]
            ) -> qutip.Qobj:
        """Create qutip.Qobj with nontrivial actions at *qubit_ids.
        op_id and qubits_id are a list of same length. op_id[j]
        acts on qubits_id[j]"""
        if len(set(qubit_ids)) < len(qubit_ids):
            raise ValueError("Duplicate atom ids in argument list.")
        if len(op_id) != len(qubit_ids):
            raise ValueError(
                "Different number of operators and qubits")
        # List of identity operators, except for op_id where requested:
        op_list = [self.op_matrix['I'] for j in range(self._size)]
        for j, qubit in enumerate(qubit_ids):
            k = self._qid_index[qubit]
            op_list[k] = self.op_matrix[op_id[j]]
        return qutip.tensor(op_list)

    def _construct_hamiltonian(self):
        def adapt(full_array):

            """Adapt list to correspond to sampling rate"""
            indexes = np.linspace(0, self._tot_duration-1,
                                  int(self.sampling_rate*self._tot_duration),
                                  dtype=int)
            return cast(np.ndarray, full_array[indexes])

        def make_vdw_term() -> float:
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

        def make_interaction_xy_term() -> float:
            """Construct the XY interaction Term.

            For each pair of qubits, calculate the distance between them,
            then assign the local operator "sigma_du * sigma_ud" at each pair.
            The units are given so that the coefficient
            includes a 1/hbar factor.
            """
            xy = 0
            # Get every pair without duplicates
            for q1, q2 in itertools.combinations(self._qdict.keys(), r=2):
                dist = np.linalg.norm(
                    self._qdict[q1] - self._qdict[q2])
                cosine = 0.
                U = 0.5 * 3700 * \
                    (1 - 3 * cosine ** 2) / dist**3
                xy += U * self._build_general_operator(
                    ['sigma_du', 'sigma_ud'], [q1, q2]
                    )
            return xy

        def build_coeffs_ops(basis, addr):

            """Build coefficients and operators for the hamiltonian QobjEvo."""
            samples = self.samples[addr][basis]
            operators = self.operators[addr][basis]
            # Choose operator names according to addressing:
            if basis == 'ground-rydberg':
                op_ids = ['sigma_gr', 'sigma_rr']
            elif basis == 'digital':
                op_ids = ['sigma_hg', 'sigma_gg']
            elif basis == 'XY':
                op_ids = ['sigma_du', 'sigma_dd']

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
        elif self.basis_name == 'XY':
            # XY Interaction Terms
            qobj_list = [make_interaction_xy_term()] if self._size > 1 else []
        else:
            # Van der Waals Interaction Terms
            qobj_list = [make_vdw_term()] if self._size > 1 else []

        # Time dependent terms:
        for addr in self.samples:
            for basis in self.samples[addr]:
                if self.samples[addr][basis]:
                    qobj_list += cast(list, build_coeffs_ops(basis, addr))

        self._times = adapt(np.arange(self._tot_duration,
                                      dtype=np.double)/1000)

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
            raise ValueError("Provided time is larger than sequence duration.")
        if time < 0:
            raise ValueError("Provided time is negative.")
        return self._hamiltonian(time/1000)  # Creates new Qutip.Qobj

    # Run Simulation Evolution using Qutip
    def run(self,
            initial_state: Optional[Union[np.ndarray, qutip.Qobj]] = None,
            progress_bar: Optional[bool] = None,
            **options: qutip.solver.Options) -> SimulationResults:
        """Simulate the sequence using QuTiP's solvers.

        Keyword Args:
            initial_state (array): The initial quantum state of the
                evolution. Will be transformed into a ``qutip.Qobj`` instance.
            progress_bar (bool): If True, the progress bar of QuTiP's
                ``qutip.sesolve()`` will be shown.
        Other Parameters:
            options: Additional simulation settings. These correspond to the
                keyword arguments of ``qutip.solver.Options`` for
                the ``qutip.sesolve()`` method.

        Returns:
            SimulationResults: Object containing the time evolution results.
        """
        if initial_state is not None:
            if isinstance(initial_state, qutip.Qobj):
                if initial_state.shape != (self.dim**self._size, 1):
                    raise ValueError("Incompatible shape of initial_state")
                self._initial_state = initial_state
            else:
                if initial_state.shape != (self.dim**self._size,):
                    raise ValueError("Incompatible shape of initial_state")
                self._initial_state = qutip.Qobj(initial_state)
        else:
            # by default, initial state is "ground" state of g-r basis.
            if self._interaction == 'XY':
                all_ground = [self.basis['d'] for _ in range(self._size)]
            else:
                all_ground = [self.basis['g'] for _ in range(self._size)]
            self._initial_state = qutip.tensor(all_ground)

        result = qutip.sesolve(self._hamiltonian,
                               self._initial_state,
                               self.eval_times,
                               progress_bar=progress_bar,
                               options=qutip.Options(max_step=5,
                                                     **options)
                               )
        meas_basis: Optional[str]
        if hasattr(self._seq, '_measurement'):
            meas_basis = cast(str, self._seq._measurement)
        else:
            meas_basis = None

        return SimulationResults(
            result.states, self.dim, self._size, self.basis_name,
            meas_basis=meas_basis
        )
