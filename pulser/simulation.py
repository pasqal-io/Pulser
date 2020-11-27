import qutip
import numpy as np
import matplotlib.pyplot as plt
import itertools

from pulser import Pulse, Sequence, Register
from pulser.devices import Chadoq2

print('simulation module...')


class Simulation:
    """
    A simulation of a pulse sequence using QuTiP
    """

    def __init__(self, sequence):
        self._seq = sequence
        self._reg = sequence._device._register
        self._L = len(self._reg.qubits)
        self._tot_duration = max([self._seq._last(ch).tf for ch in self._seq._schedule])
        self._times = np.arange(self._tot_duration, dtype=np.double)
        self.addressing = {'global_rydberg' : False,
                           'local_rydberg' : False,
                           'global_raman' : False,
                           'local_raman' : False}

        self.active_qubits = self._get_active_qubits()
        self._qid_index = {qid: i for i, qid in enumerate(self._reg.qubits)}

        self._extract_samples()

        self._H = self._construct_hamiltonian()

    def _get_active_qubits(self):
        active=[]
        for basis in self._seq._last_used:
            for qid,time in self._seq._last_used[basis].items():
                if time > 0 and qid not in active:
                    active.append(qid)
        return active

    def _extract_samples(self):
        def _record_samples(channel, samples_dict):
            for slot in self._seq._schedule[channel]:
                if isinstance(slot.type, Pulse):
                    for qubit in slot.targets: # May be more than one in the future
                        samples_dict[qubit]['amp'][slot.ti:slot.tf] += slot.type.amplitude.samples
                        samples_dict[qubit]['det'][slot.ti:slot.tf] += slot.type.detuning.samples
                        samples_dict[qubit]['phase'][slot.ti:slot.tf] = slot.type.phase

        def _make_sample_dict():
            #Duration includes retargeting, delays, etc.
            return {'amp': np.zeros(self._tot_duration),
                    'det': np.zeros(self._tot_duration),
                    'phase': np.zeros(self._tot_duration)}

        for channel in self._seq.declared_channels:
            channel_addressing = self._seq.declared_channels[channel].addressing
            channel_basis = self._seq.declared_channels[channel].basis

            if channel_addressing == 'Global':
                if channel_basis == 'ground-rydberg':
                    # Define the samples dictionary the first time
                    self.addressing['global_rydberg'] = True
                    if not hasattr(self, '_global_rydberg_samples'):
                        self._global_rydberg_samples = _make_sample_dict()
                    _record_samples(channel, self._global_rydberg_samples)

                if channel_basis == 'digital':
                    self.addressing['global_raman'] = True
                    if not hasattr(self, '_global_raman_samples'):
                        self._global_raman_samples = _make_sample_dict()
                    _record_samples(channel, self._global_raman_samples)

            if channel_addressing == 'Local':
                if channel_basis == 'ground-rydberg':
                    self.addressing['local_rydberg'] = True
                    if not hasattr(self, '_local_rydberg_samples'):
                        self._local_rydberg_samples = {qubit : _make_sample_dict() for qubit in self.active_qubits}
                    _record_samples(channel, self._local_rydberg_samples)

                if channel_basis == 'digital':
                    self.addressing['local_raman'] = True
                    if not hasattr(self, '_local_raman_samples'):
                        self._local_raman_samples = {qubit : _make_sample_dict() for qubit in self.active_qubits}
                    _record_samples(channel, self._local_raman_samples)

    def _create_basis_and_operators(self):
        """Creates the basis elements"""
        if self.basis_name == 'all':
            self._dim = 3
            self._basis = {'r': qutip.basis(3, 0),
                           'g': qutip.basis(3, 1),
                           'h': qutip.basis(3, 2)
                           }
            self._operators = {'I': qutip.qeye(3),
                               'sigma_gr': self._basis['g'] * self._basis['r'].dag(),
                               'sigma_hg': self._basis['h'] * self._basis['g'].dag(),
                               'sigma_rr': self._basis['r'] * self._basis['r'].dag(),
                               'sigma_gg': self._basis['g'] * self._basis['g'].dag(),
                               'sigma_hh': self._basis['h'] * self._basis['h'].dag(),
                               }
        elif self.basis_name == 'rydberg':
            self._dim = 2
            self._basis = {'r': qutip.basis(2, 0),
                           'g': qutip.basis(2, 1)
                           }
            self._operators = {'I': qutip.qeye(2),
                               'sigma_gr': self._basis['g'] * self._basis['r'].dag(),
                               'sigma_rr': self._basis['r'] * self._basis['r'].dag(),
                               'sigma_gg': self._basis['g'] * self._basis['g'].dag()
                               }
        elif self.basis_name == 'digital':
            self._dim = 2
            self._basis = {'g': qutip.basis(2, 0),
                           'h': qutip.basis(2, 1)
                           }
            self._operators = {'I': qutip.qeye(2),
                               'sigma_hg': self._basis['h'] * self._basis['g'].dag(),
                               'sigma_hh': self._basis['h'] * self._basis['h'].dag(),
                               'sigma_gg': self._basis['g'] * self._basis['g'].dag()
                               }


    # Define Operators

    def _build_local_operator(self, operator, *qubit_ids):
        """
        Returns a local gate at atoms in indexes corresponding to qubit_id
        """
        if len(set(qubit_ids)) < len(qubit_ids):
            raise ValueError("Duplicate atom ids in argument list")

        temp = [qutip.qeye(operator.shape[0]) for _ in range(self._L)] #List of identities with shape of operator
        for qubit in qubit_ids:
            temp[self._qid_index[qubit]] = operator
        return qutip.tensor(temp)

    def _build_array_local_operators(self,operator):
        """
        Returns an array with 'operator' acting nontrivially on each qubit index.
        We create local operators only for the qubits that are active during the
        evolution. If all are active, then the sum in self._global_operators can
        be calculated.
        """
        op_array = [ self._build_local_operator(operator,qubit) for qubit in self.active_qubits ]
        return op_array

    def _build_operator_array_dict(self):
        return {key: self._build_array_local_operators(op)
                     for key, op in self._operators.items() if key != 'I'}

    def _make_VdW_term(self):
        """
        Constructs the Van der Waals Interaction Terms
        """
        VdW = 0
        # Get every pair without duplicates
        for qubit1, qubit2 in itertools.combinations(self._reg._ids, r=2):
            R = np.linalg.norm(self._reg.qubits[qubit1] - self._reg.qubits[qubit2])
            coeff = 1e6 / (R**6)
            VdW += coeff * 0.5 * self._build_local_operator(
                self._operators['sigma_rr'], qubit1, qubit2)
        return VdW

    def _make_rotation_term(self, basis, qubit=None, local_samples=None, global_samples=None):
        """
        Creates the terms that go into the Hamiltonian. Note that this works
        once a basis has been defined.
        For the moment one will not have both local and global samples given
        (see _construct_hamiltonian method)
        """
        # Choose operator names according to addressing:
        if basis == 'rydberg':
            xy_name = "sigma_gr"
            z_name = "sigma_rr"
        elif basis == 'raman':
            xy_name = "sigma_hg"
            z_name = "sigma_gg"

        # Create Coefficients and Operators:
        if global_samples:
            xy_coeff = global_samples['amp']*np.exp(-1j*global_samples['phase'])
            z_coeff = global_samples['det']
            if not hasattr(self, _global_array_operators):
                self._global_operators = {
                                        xy_name: sum(self._array_operators[xy_name]),
                                        z_name: sum(self._array_operators[z_name])
                                         }
            xy_operator = self._global_operators[xy_name]
            z_operator = self._global_operators[z_name]

        elif local_samples:
            xy_coeff = local_samples['amp']*np.exp(-1j*local_samples['phase'])
            z_coeff = local_samples['det']
            qubit_index = self._qid_index[qubit]
            xy_operator = self._array_operators[xy_name][qubit_index]
            z_operator = self._array_operators[z_name][qubit_index]

        op_list = []
        if np.any(xy_coeff != 0):
            op_list.append([xy_operator, xy_coeff])
        if np.any(z_coeff != 0):
            op_list.append([0.5 * z_operator, z_coeff])

        # Build rotation term as QObjEvo :
        if op_list:
            rotation_term = qutip.QobjEvo(op_list, tlist=self._times)
        else:
            rotation_term = 0

        return rotation_term

    def _construct_hamiltonian(self):

        # Decide appropriate basis:
        if not self.addressing['global_raman'] and not self.addressing['local_raman']:
            self.basis_name = 'rydberg'
        elif not self.addressing['global_rydberg'] and not self.addressing['local_rydberg']:
            self.basis_name = 'raman'
        else:
            self.basis_name = 'all' # All three states

        # Construct Basis, basic operators and array of local operators
        self._create_basis_and_operators()
        self._array_operators = self._build_operator_array_dict()

        # Construct Hamiltonian

        # Time independent term:
        if self.basis_name == 'digital':
            H = 0
        else:
            # Van der Waals Interaction Terms
            H = qutip.QobjEvo( [self._make_VdW_term()] )

        # Time dependent terms:
        if self.addressing["global_rydberg"]:
            H += self._make_rotation_term('rydberg',
                                            global_samples = self._global_ryd_samples
                                         )
        if self.addressing["global_raman"]:
            H += self._make_rotation_term('raman',
                                            global_samples = self._global_raman_samples
                                            )

        if self.addressing["local_rydberg"]:
            for qubit in self.active_qubits:
                H += self._make_rotation_term('rydberg',
                                                qubit,
                                                local_samples = self._local_rydberg_samples[qubit]
                                             )
        if self.addressing["local_raman"]:
            for qubit in self.active_qubits:
                H += self._make_rotation_term('raman',
                                                qubit,
                                                local_samples = self._local_raman_samples[qubit]
                                             )
        return H + H.dag()

    # Run Simulation Evolution using Qutip
    def run(self, initial_state=None, observable=None, plot=True, all_states=False, custom_label=None):
        """
        Simulates the sequence with the predefined observable
        """
        if initial_state:
            self._initial_state = initial_state
        else:
            # by default, lowest energy state
            self._initial_state = qutip.tensor([qutip.basis(self._dim,self._dim - 1) for _ in range(self._L)])

        if observable:
            # With observables, we get their expectation value
            result = qutip.sesolve(self._H, self._initial_state, self._times, [observable])
            self.output = result.expect[0]
            if plot:
                plt.plot(self._times, self.output, label=custom_label)
                plt.legend()
        else:
            # Without observables, we get the output state
            result = qutip.sesolve(self._H, self._initial_state, self._times)
            if all_states:
                self.output = result.states # Get all states of evolution
            else:
                self.output = result.states[-1] # Get only final state of evolution
