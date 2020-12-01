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
        self.active_local = self._get_active()
        self._qid_index = {qid: i for i, qid in enumerate(self._reg.qubits)}
        self._active_qid_index = {
            basis: {qid: i for i, qid in enumerate(self.active_local[basis])}
            for basis in ['ground-rydberg', 'digital']}

        self.dim = 3  # Default value
        self.basis = {}
        self.operators = {}
        self._array_operators = {basis: []
                                 for basis in ['ground-rydberg', 'digital']}

        self._extract_samples()
        self._decide_basis()
        self._create_basis_and_operators()
        self._hamiltonian = self._construct_hamiltonian()

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

    def _get_active(self):
        active = {basis: set() for basis in ['ground-rydberg', 'digital']}
        for channel in self._seq.declared_channels:
            basis = self._seq.declared_channels[channel].basis
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

    def _build_array_operators(self, basis):
        """
        Return array with 'operator' acting non-trivially on each active qid.

        We create local operators only for the qubits that are active
        during the evolution. If all are active, then the sum in
        self._global_operators can be calculated.
        """
        op_dic = {key: [self._build_local_operator(op, qubit)
                        for qubit in self.active_local[basis]]
                  for key, op in self.operators.items() if key != 'I'}

        return op_dic

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
            return vdw

        def make_coeffs_ops(basis, addressing):
            # Choose operator names according to addressing:
            if basis == 'ground-rydberg':
                xy_name = "sigma_gr"
                z_name = "sigma_rr"
            elif basis == 'digital':
                xy_name = "sigma_hg"
                z_name = "sigma_gg"

            if addressing == 'Global':
                samples = self.samples[addressing][basis]
                xy_coeff = samples['amp'] * np.exp(-1j * samples['phase'])
                z_coeff = samples['det']
                xy_op = sum(self._array_operators[basis][xy_name])
                z_op = sum(self._array_operators[basis][z_name])
            elif addressing == 'Local':
                for qubit in self.active_local[basis]:
                    q_index = self._active_qid_index[basis][qubit]
                    samples = self.samples[addressing][basis][qubit]
                    xy_coeff = samples['amp'] * np.exp(-1j * samples['phase'])
                    z_coeff = samples['det']
                    xy_op = self._array_operators[basis][xy_name][q_index]
                    z_op = self._array_operators[basis][z_name][q_index]

            return [xy_coeff, xy_op, z_coeff, z_op]

        def make_rotation_term(blocks):
            """
            Create the terms that build the Hamiltonian.

            Note that this works once a basis has been defined.
            """
            xy_coeff, xy_op, z_coeff, z_op = blocks
            op_list = []
            if np.any(xy_coeff != 0):
                op_list.append([xy_op, xy_coeff])
            if np.any(z_coeff != 0):
                op_list.append([0.5 * z_op, z_coeff])

            # Build rotation term as QObjEvo :
            if op_list:
                rotation_term = qutip.QobjEvo(op_list, tlist=self._times)
            else:
                rotation_term = 0

            return rotation_term

        # Time independent term:
        if self.basis_name == 'digital':
            ham = 0
        else:
            # Van der Waals Interaction Terms
            ham = qutip.QobjEvo([make_vdw_term()])

        # Time dependent terms:
        for addr in self.addressing:
            for basis in self.addressing[addr]:
                if not self._array_operators[basis]:
                    self._array_operators[basis] = \
                                            self._build_array_operators(basis)
                if self.samples[addr][basis]:
                    blocks = make_coeffs_ops(basis, addr)
                    ham += make_rotation_term(blocks)

        ham = ham + ham.dag()
        ham.compress()

        return ham

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
            psi0 = qutip.tensor(
                        [qutip.basis(self.dim, self.dim - 1)
                            for _ in range(self._size)]
                        )

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
