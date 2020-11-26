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

        self._define_sample_dicts()
        self._extract_samples()

        self._H = self._construct_hamiltonian()

    def _make_sample_dict(self):
        return {'amp': np.zeros(self._tot_duration), #Duration includes retargeting, delays, etc.
                'det': np.zeros(self._tot_duration),
                'phase': np.zeros(self._tot_duration)}

    def _define_sample_dicts(self):

        self._global_rydberg_samples = self._make_sample_dict()
        self._global_raman_samples = self._make_sample_dict()

        self._local_ryd_samples = {}
        for qubit in self._reg.qubits:
            self._local_ryd_samples[qubit] = self._make_sample_dict()

        self._local_raman_samples = {}
        for qubit in self._reg.qubits:
            self._local_raman_samples[qubit] = self._make_sample_dict()

    def _record_samples_from_slot(self, slot, samples_dict):
        samples_dict['amp'][slot.ti:slot.tf] += slot.type.amplitude.samples
        samples_dict['det'][slot.ti:slot.tf] += slot.type.detuning.samples
        samples_dict['phase'][slot.ti:slot.tf] = slot.type.phase,

    def _extract_samples(self):

        for channel in self._seq.declared_channels:
            addressing_name = self._seq.declared_channels[channel].addressing
            basis_name = self._seq.declared_channels[channel].basis

            if addressing_name == 'Global':
                if basis_name == 'ground-rydberg':
                    self.addressing['global_rydberg'] = True
                    for slot in self._seq._schedule[channel]:
                        if isinstance(slot.type, Pulse):
                            self._record_samples_from_slot(slot, self._global_rydberg_samples)

                if basis_name == 'digital':
                    self.addressing['global_raman'] = True
                    for slot in self._seq._schedule[channel]:
                        if isinstance(slot.type, Pulse):
                            self._record_samples_from_slot(slot, self._global_raman_samples)

            if addressing_name == 'Local':
                if basis_name == 'ground-rydberg':
                    self.addressing['local_rydberg'] = True
                    for slot in self._seq._schedule[channel]:
                        if isinstance(slot.type, Pulse):
                            # Get the target from set object
                            qubit = list(slot.targets)[0]
                            self._record_samples_from_slot(slot, self._local_ryd_samples[qubit])

                if basis_name == 'digital':
                    self.addressing['local_raman'] = True
                    for slot in self._seq._schedule[channel]:
                        if isinstance(slot.type, Pulse):
                            # Get the target from set object
                            qubit = list(slot.targets)[0]
                            self._record_samples_from_slot(slot, self._local_raman_samples[qubit])


    def _create_basis(self,basis):
        """Creates the basis elements"""
        if basis == 'all':
            self._dim = 3
            self._basis = {'r': qutip.basis(3, 0),
                           'g': qutip.basis(3, 1),
                           'h': qutip.basis(3, 2)
                           }
        if basis == 'rydberg':
            self._dim = 2
            self._basis = {'r': qutip.basis(2, 0),
                           'g': qutip.basis(2, 1)
                           }
        if basis == 'digital':
            self._dim = 2
            self._basis = {'g': qutip.basis(2, 0),
                           'h': qutip.basis(2, 1)
                           }

    def _create_operators(self,basis):

        if basis == 'all':
            r = self._basis['r']
            g = self._basis['g']
            h = self._basis['h']
            self._operators = {'I': qutip.qeye(3),
                               'sigma_gr': g * r.dag(),
                               'sigma_hg': h * g.dag(),
                               'sigma_rr': r * r.dag(),
                               'sigma_gg': g * g.dag(),
                               'sigma_hh': h * h.dag()
                               }

        if basis == 'rydberg':
            r = self._basis['r']
            g = self._basis['g']
            self._operators = {'I': qutip.qeye(2),
                               'sigma_gr': g * r.dag(),
                               'sigma_rr': r * r.dag(),
                               'sigma_gg': g * g.dag()
                               }

        if basis == 'digital':
            g = self._basis['g']
            h = self._basis['h']
            self._operators = {'I': qutip.qeye(2),
                               'sigma_hg': h * g.dag(),
                               'sigma_hh': h * h.dag(),
                               'sigma_gg': g * g.dag()
                               }

    # Define Operators

    def _build_local_operator(self, operator, *qubit_ids):
        """
        Returns a local gate at atoms in positions given by qubit_id
        """
        if len(set(qubit_ids)) < len(qubit_ids):
            raise ValueError("Duplicate atom ids in argument list")

        temp = [qutip.qeye(operator.shape[0]) for _ in range(self._L)] #List of identities with shape of operator
        for qubit_id in qubit_ids:
            pos = list(self._reg.qubits).index(qubit_id)
            temp[pos] = operator
        return qutip.tensor(temp)

    def _build_array_local_operators(self,operator):
        """
        Returns an array with 'operator' acting nontrivially on each qubit position
        TO DO: If there's no global terms, optimize for only the qubits being addressed
        """
        op_array = [ self._build_local_operator(operator,qubit) for qubit in self._reg.qubits ]
        return op_array

    def _build_operator_array_dict(self):

        if self.addressing['local_raman'] or self.addressing['global_raman']:
            if self.addressing['global_rydberg'] or self.addressing['local_rydberg']:
                return {
                'sigma_gg' : self._build_array_local_operators(self._operators['sigma_gg']),
                'sigma_gr' : self._build_array_local_operators(self._operators['sigma_gr']),
                'sigma_rr' : self._build_array_local_operators(self._operators['sigma_rr']),
                'sigma_hg' : self._build_array_local_operators(self._operators['sigma_hg']),
                'sigma_hh' : self._build_array_local_operators(self._operators['sigma_hh'])
                }
            else: # Only digital basis
                return {
                'sigma_hg' : self._build_array_local_operators(self._operators['sigma_hg']),
                'sigma_hh' : self._build_array_local_operators(self._operators['sigma_hh'])
                }
        else: # Only rydberg basis
            return {
            'sigma_gr' : self._build_array_local_operators(self._operators['sigma_gr']),
            'sigma_rr' : self._build_array_local_operators(self._operators['sigma_rr'])
            }

    def _make_VdW_term(self):
        """
        Constructs the Van der Waals Interaction Terms
        """
        VdW = 0
        # Get every pair without duplicates
        for qubit1, qubit2 in itertools.combinations(self._reg._ids, r=2):
            R = np.sqrt(
                np.sum((self._reg.qubits[qubit1] - self._reg.qubits[qubit2])**2))
            coeff = 1e6 / (R**6)
            VdW += coeff * 0.5 * self._build_local_operator(
                self._operators['sigma_rr'], qubit1, qubit2)
        return VdW

    def _make_rotation_term(self, addressing, qubit=None, local_samples=None, global_samples=None):
        """
        Creates the terms that go into the Hamiltonian. Note that this works
        once a basis has been defined.
        Notice for the moment one will not have both local and global samples given
        (see _construct_hamiltonian method)
        """
        # Choose operator names according to addressing:
        if addressing == 'rydberg':
            xy_name = "sigma_gr"
            z_name = "sigma_rr"
        else: #  addressing == 'raman'
            xy_name = "sigma_hg"
            z_name = "sigma_gg"

        # Create Coefficients and Operators:
        if global_samples:
            global_xy_coeff = global_samples['amp'] * np.exp(-1j * global_samples['phase'])
            if np.abs(sum(global_xy_coeff)) > 0: global_xy_operator = sum(self._array_operators[xy_name])

            global_z_coeff = global_samples['det']
            if np.abs(sum(global_z_coeff)) > 0:  global_z_operator = sum(self._array_operators[z_name])

        if local_samples:
            xy_coeff = local_samples['amp'] * np.exp(-1j * local_samples['phase'])
            z_coeff = local_samples['det']

            qubit_pos = list(self._reg.qubits).index(qubit) # Get qubit position
            xy_operator = self._array_operators[xy_name][qubit_pos]
            z_operator = self._array_operators[z_name][qubit_pos]

        # Build rotation term as QObjEvo :
        if global_samples and not local_samples:
            op_list = []
            if np.abs(sum(global_xy_coeff)) > 0: op_list.append([global_xy_operator, gloabl_xy_coeff])
            if np.abs(sum(global_z_coeff)) > 0: op_list.append([0.5 * global_z_operator, global_z_coeff])
            if len(op_list) > 0:
                rotation_term = qutip.QobjEvo( op_list, tlist = self._times )
            else:
                rotation_term = 0

        if not global_samples and local_samples:
            op_list = []
            if np.abs(sum(xy_coeff)) > 0: op_list.append([xy_operator, xy_coeff])
            if np.abs(sum(z_coeff)) > 0: op_list.append([0.5 * z_operator, z_coeff])
            if len(op_list) > 0:
                rotation_term = qutip.QobjEvo( op_list, tlist = self._times )
            else:
                rotation_term = 0

        return rotation_term


    def _construct_hamiltonian(self):

        # Decide appropriate basis:
        if not self.addressing['global_raman'] and not self.addressing['local_raman']:
            basis_name = 'digital'
        elif not self.addressing['global_rydberg'] and not self.addressing['local_rydberg']:
            basis_name = 'rydberg'
        else:
            basis_name = 'all' # All three states


        # Construct Basis and array of operators
        self._create_basis(basis_name)
        self._create_operators(basis_name)
        self._array_operators = self._build_operator_array_dict()

        # Construct Hamiltonian

        # Time independent term:
        if basis_name == 'digital':
            H = 0
        else:
            # Van der Waals Interaction Terms
            H = qutip.QobjEvo( [self._make_VdW_term()] )

        # Time dependent terms:
        if self.addressing["global_rydberg"]:
            H += self._make_rotation_term('rydberg',
                                            global_samples = self._global_ryd_samples
                                         )
        if self.addressing["local_rydberg"]:
            for qubit in self._reg.qubits:
                H += self._make_rotation_term('rydberg',
                                                qubit,
                                                local_samples = self._local_ryd_samples[qubit]
                                             )

        if self.addressing["global_raman"]:
            H += self._make_rotation_term('raman',
                                            global_samples = self._global_raman_samples
                                         )
        if self.addressing["local_raman"]:
            for qubit in self._reg.qubits:
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
