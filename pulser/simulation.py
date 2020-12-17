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

import qutip
import numpy as np
from copy import deepcopy

from pulser import Pulse, Sequence


class Simulation:
    """Simulation of a pulse sequence using QuTiP.

    Creates a Hamiltonian object with the proper dimension according to the
    pulse sequence given, then provides a method to time-evolve an initial
    state using the QuTiP solvers.

    Args:
        sequence (pulser.Sequence): An instance of a Pulser Sequence that we
                                    want to simulate.
    """

    def __init__(self, sequence):
        """Initialize the Simulation with a specific pulser.Sequence."""
        if not isinstance(sequence, Sequence):
            raise TypeError("The provided sequence has to be a valid "
                            "pulser.Sequence instance.")
        self._seq = sequence
        self._reg = sequence._register
        self._size = len(self._reg.qubits)
        self._tot_duration = max(
                        [self._seq._last(ch).tf for ch in self._seq._schedule]
                                )
        self._times = np.arange(self._tot_duration, dtype=np.double)
        self._qid_index = {qid: i for i, qid in enumerate(self._reg.qubits)}

        self.samples = {addr: {basis: {}
                               for basis in ['ground-rydberg', 'digital']}
                        for addr in ['Global', 'Local']}
        self.operators = deepcopy(self.samples)

        self._extract_samples()
        self._build_basis_and_op_matrices()
        self._construct_hamiltonian()

    def _extract_samples(self):
        """Populate samples dictionary with every pulse in the sequence."""

        def prepare_dict():
            # Duration includes retargeting, delays, etc.
            return {'amp': np.zeros(self._tot_duration),
                    'det': np.zeros(self._tot_duration),
                    'phase': np.zeros(self._tot_duration)}

        def write_samples(slot, samples_dict):
            samples_dict['amp'][slot.ti:slot.tf] += slot.type.amplitude.samples
            samples_dict['det'][slot.ti:slot.tf] += slot.type.detuning.samples
            samples_dict['phase'][slot.ti:slot.tf] = slot.type.phase

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
            projectors = ['gr', 'hg', 'rr', 'gg', 'hh']

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
                       for q_id in self._reg.qubits)

        if len(set(qubit_ids)) < len(qubit_ids):
            raise ValueError("Duplicate atom ids in argument list.")
        # List of identity matrices with shape of operator:
        temp = [self.op_matrix['I'] for _ in range(self._size)]
        for q_id in qubit_ids:
            temp[self._qid_index[q_id]] = self.op_matrix[op_id]
        return qutip.tensor(temp)

    def _construct_hamiltonian(self):
        def make_vdw_term():
            """Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate the distance between them, then
            assign the local operator "sigma_rr" at each pair. The units are
            given so that the coefficient includes a 1/hbar factor.
            """
            vdw = 0
            # Get every pair without duplicates
            for qubit1, qubit2 in itertools.combinations(self._reg._ids, r=2):
                dist = np.linalg.norm(
                        self._reg.qubits[qubit1] - self._reg.qubits[qubit2])
                U = 0.5 * 5.008e6 / dist**6  # = U/hbar
                self._U = 2*U
                vdw += U * self._build_operator('sigma_rr', qubit1, qubit2)
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
                for coeff, op_id in zip(coeffs, op_ids):
                    if np.any(coeff != 0):
                        # Build once global operators as they are needed
                        if op_id not in operators:
                            operators[op_id] =\
                                    self._build_operator(op_id, global_op=True)
                        terms.append([operators[op_id], coeff])
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
                                          coeff])

            self.operators[addr][basis] = operators
            return terms

        # Time independent term:
        if self.basis_name == 'digital':
            qobj_list = []
        else:
            # Van der Waals Interaction Terms
            qobj_list = [make_vdw_term()]

        # Time dependent terms:
        for addr in self.samples:
            for basis in self.samples[addr]:
                if self.samples[addr][basis]:
                    qobj_list += build_coeffs_ops(basis, addr)

        ham = qutip.QobjEvo(qobj_list, tlist=self._times/1000)
        ham = ham + ham.dag()
        ham.compress()
        self._hamiltonian = ham

    # Run Simulation Evolution using Qutip
    def run(self, initial_state=None, obs_list=None, progress_bar=None):
        """Simulate the sequence using QuTiP's solvers.

        Keyword Args:
            initial_state (qutip.Qobj): The initial quantum state of the
                           evolution.
            obs_list (list): A list of qutip.Qobj observables whose
                      expectation value will be calculated.
        """
        if initial_state:
            psi0 = initial_state
        else:
            # by default, initial state is "ground" state of g-r basis.
            all_ground = [self.basis['g'] for _ in range(self._size)]
            psi0 = qutip.tensor(all_ground)

        if obs_list:
            if not isinstance(obs_list, list):
                raise TypeError("`obs_list` must be a list of operators")
            print('Observables provided. Calculating expectation value...')
            result = qutip.sesolve(self._hamiltonian,
                                   psi0,
                                   self._times/1000,
                                   obs_list,
                                   progress_bar=progress_bar,
                                   options=qutip.Options(max_step=5,
                                                         nsteps=20000)
                                   )
        else:
            print('No observable provided. Calculating state evolution...')
            result = qutip.sesolve(self._hamiltonian,
                                   psi0,
                                   self._times/1000,
                                   progress_bar=progress_bar,
                                   options=qutip.Options(max_step=5,
                                                         nsteps=20000)
                                   )
        self.output = result
