import itertools

import qutip
import numpy as np
import matplotlib.pyplot as plt

from pulser import Pulse

print('simulation module...')


class Simulation:
    """Simulation of a pulse sequence using QuTiP."""

    def __init__(self, sequence):
        self._seq = sequence
        self._reg = sequence._device._register
        self._size = len(self._reg.qubits)
        self._tot_duration = max(
                        [self._seq._last(ch).tf for ch in self._seq._schedule]
                                )
        self._times = np.arange(self._tot_duration, dtype=np.double)

        self.samples = {addr: {basis: {}
                               for basis in ['ground-rydberg', 'digital']}
                        for addr in ['Global', 'Local']}
        self.addressing = {addr: {basis: False
                                  for basis in ['ground-rydberg', 'digital']}
                           for addr in ['Global', 'Local']}

        # self.active_local = self._get_active_local()
        self.active_local = self._get_active_local()
        self._qid_index = {qid: i for i, qid in enumerate(self._reg.qubits)}
        self._active_qid_index = {basis: {qid: self._qid_index[qid]
                                          for qid in self.active_local[basis]}
                                  for basis in ['ground-rydberg', 'digital']}

        self.dim = 3  # Default value
        self.basis = {}
        self.operators = {}
        self._local_operators = {basis: {qubit: {}
                                 for qubit in self.active_local[basis]}
                                 for basis in ['ground-rydberg', 'digital']}
        self._global_operators = {basis: {}
                                  for basis in ['ground-rydberg', 'digital']}

        self._extract_samples()
        self._decide_basis()
        self._create_basis_and_operators()
        self._construct_hamiltonian()

    def _get_active_local(self):
        active = {basis: set() for basis in ['ground-rydberg', 'digital']}
        for channel in self._seq.declared_channels:
            addr = self._seq.declared_channels[channel].addressing
            basis = self._seq.declared_channels[channel].basis
            if addr == 'Local':
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        active[basis] |= slot.targets
        return active

    def _extract_samples(self):

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
            ch_addr = self._seq.declared_channels[channel].addressing
            ch_basis = self._seq.declared_channels[channel].basis
            self.addressing[ch_addr][ch_basis] = True

            if ch_addr == 'Global':
                if not self.samples[ch_addr][ch_basis]:
                    self.samples[ch_addr][ch_basis] = prepare_dict()
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        samples_dict = self.samples[ch_addr][ch_basis]
                        write_samples(slot, samples_dict)

            elif ch_addr == 'Local':
                if not self.samples[ch_addr][ch_basis]:
                    self.samples[ch_addr][ch_basis] = \
                                    {qubit: prepare_dict()
                                     for qubit in self.active_local[ch_basis]}
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        for qubit in slot.targets:  # Allow multiaddressing
                            samples_dict = self.samples[ch_addr][ch_basis]
                            write_samples(slot, samples_dict[qubit])

    def _decide_basis(self):
        """Decide appropriate basis."""
        if not self.addressing['Global']['digital']\
                and not self.addressing['Local']['digital']:
            self.basis_name = 'ground-rydberg'
        elif not self.addressing['Global']['ground-rydberg']\
                and not self.addressing['Local']['ground-rydberg']:
            self.basis_name = 'digital'
        else:
            self.basis_name = 'all'  # All three states

    def _create_basis_and_operators(self):
        """Create the basis elements."""
        if self.basis_name == 'all':
            self.dim = 3
            self.basis = {'r': qutip.basis(3, 0),
                          'g': qutip.basis(3, 1),
                          'h': qutip.basis(3, 2)
                          }
            self.operators = {
                    'I': qutip.qeye(3),
                    'sigma_gr': self.basis['g'] * self.basis['r'].dag(),
                    'sigma_hg': self.basis['h'] * self.basis['g'].dag(),
                    'sigma_rr': self.basis['r'] * self.basis['r'].dag(),
                    'sigma_gg': self.basis['g'] * self.basis['g'].dag(),
                    'sigma_hh': self.basis['h'] * self.basis['h'].dag()
                               }
        elif self.basis_name == 'ground-rydberg':
            self.dim = 2
            self.basis = {'r': qutip.basis(2, 0),
                          'g': qutip.basis(2, 1)
                          }
            self.operators = {
                    'I': qutip.qeye(2),
                    'sigma_gr': self.basis['g'] * self.basis['r'].dag(),
                    'sigma_rr': self.basis['r'] * self.basis['r'].dag(),
                    'sigma_gg': self.basis['g'] * self.basis['g'].dag()
                               }
        elif self.basis_name == 'digital':
            self.dim = 2
            self.basis = {'g': qutip.basis(2, 0),
                          'h': qutip.basis(2, 1)
                          }
            self.operators = {
                    'I': qutip.qeye(2),
                    'sigma_hg': self.basis['h'] * self.basis['g'].dag(),
                    'sigma_hh': self.basis['h'] * self.basis['h'].dag(),
                    'sigma_gg': self.basis['g'] * self.basis['g'].dag()
                              }

    def _build_local_operator(self, operator, *qubit_ids):
        """Return local gate in indexes corresponding to qubit_id."""
        if len(set(qubit_ids)) < len(qubit_ids):
            raise ValueError("Duplicate atom ids in argument list.")

        # List of identity matrices with shape of operator
        temp = [qutip.qeye(operator.shape[0]) for _ in range(self._size)]
        for qubit in qubit_ids:
            temp[self._qid_index[qubit]] = operator
        return qutip.tensor(temp)

    def _build_global_operator(self, op_name):
        return sum([self._build_local_operator(self.operators[op_name], qubit)
                    for qubit in self._reg.qubits])

    def _construct_hamiltonian(self):
        def make_vdw_term():
            """
            Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate each distance pairwise, then
            assign the local operator "sigma_rr" on each pair.
            """
            vdw = 0
            # Get every pair without duplicates
            for qubit1, qubit2 in itertools.combinations(self._reg._ids, r=2):
                dist = np.linalg.norm(
                        self._reg.qubits[qubit1] - self._reg.qubits[qubit2])
                vdw += (1e6 / (dist**6)) * 0.5 * \
                    self._build_local_operator(self.operators['sigma_rr'],
                                               qubit1, qubit2)
            return [vdw]

        def build_coeffs_ops(basis, addressing):
            # Choose operator names according to addressing:
            if basis == 'ground-rydberg':
                op_ids = ['sigma_gr', 'sigma_rr']
            elif basis == 'digital':
                op_ids = ['sigma_hg', 'sigma_gg']

            terms = []
            if addressing == 'Global':
                samples = self.samples[addressing][basis]
                coeffs = [samples['amp'] * np.exp(-1j * samples['phase']),
                          samples['det']]
                for coeff, op_id in zip(coeffs, op_ids):
                    if np.any(coeff != 0):
                        # Build once global operators as they are needed
                        if not self._global_operators[basis] \
                         or not self._global_operators[basis][op_id]:
                            self._global_operators[basis][op_id] = \
                             self._build_global_operator(op_id)
                        terms.append([self._global_operators[basis][op_id],
                                      coeff])
            elif addressing == 'Local':
                for qubit in self.active_local[basis]:
                    samples = self.samples[addressing][basis][qubit]
                    coeffs = [samples['amp'] * np.exp(-1j * samples['phase']),
                              samples['det']]
                    for coeff, op_id in zip(coeffs, op_ids):
                        if np.any(coeff != 0):
                            if not self._local_operators[basis][qubit]:
                                self._local_operators[basis][qubit][op_id] = \
                                 self._build_local_operator(
                                    self.operators[op_id], qubit)
                            terms.append(
                                [self._local_operators[basis][qubit][op_id],
                                 coeff])
            return terms

        # Time independent term:
        if self.basis_name == 'digital':
            qobj_list = []
        else:
            # Van der Waals Interaction Terms
            qobj_list = make_vdw_term()

        # Time dependent terms:
        for addr in self.addressing:
            for basis in self.addressing[addr]:
                if self.samples[addr][basis]:
                    qobj_list += build_coeffs_ops(basis, addr)

        ham = qutip.QobjEvo(qobj_list, tlist=self._times)
        ham = ham + ham.dag()
        ham.compress()
        self._hamiltonian = ham

    # Run Simulation Evolution using Qutip
    def run(self, initial_state=None, observable=None, plot=True,
            all_states=False, custom_label=None):
        """
        Simulate the sequence.

        Can either give a predefined observable, or the final state after
        evolution, or the list of all states during evolution. If plot,
        optionally add a custom label
        """
        if initial_state:
            psi0 = initial_state
        else:
            # by default, "all down" state
            all_down = [qutip.basis(self.dim, self.dim - 1)
                        for _ in range(self._size)]
            psi0 = qutip.tensor(all_down)

        if observable:
            # With observables, we get their expectation value
            result = qutip.sesolve(self._hamiltonian, psi0,
                                   self._times, [observable])
            self.output = result.expect[0]
            if plot:
                plt.plot(self._times, self.output, label=custom_label)
                plt.legend()
        else:
            # Without observables, we get the output state
            result = qutip.sesolve(self._hamiltonian, psi0,
                                   self._times)
            if all_states:
                self.output = result.states  # All states of evolution
            else:
                self.output = result.states[-1]  # Final state of evolution
