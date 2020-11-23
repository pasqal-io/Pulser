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
        self._tot_duration = max(
            [self._seq._last(ch).tf for ch in self._seq._schedule])
        self._times = np.arange(self._tot_duration, dtype=np.double)
        self.addressing = {'global' : False, 'local_ryd' : False, 'digital' : False}

        self._define_sample_dicts()
        self._extract_samples()

        self._construct_hamiltonian()

        self._initial_state = qutip.tensor(
            [qutip.basis(3, 0) for _ in range(self._L)])

    def _make_sample_dict(self):
        return {'amp': np.zeros(self._tot_duration),
                'det': np.zeros(self._tot_duration),
                'phase': np.zeros(self._tot_duration)}

    def _define_sample_dicts(self):
        self._global_samples = self._make_sample_dict()

        self._local_ryd_samples = {}
        for qubit in self._reg.qubits:
            self._local_ryd_samples[qubit] = self._make_sample_dict()

        self._local_raman_samples = {}
        for qubit in self._reg.qubits:
            self._local_raman_samples[qubit] = self._make_sample_dict()

    def _record_samples_from_slot(self, slot, dictionary):
        dictionary['amp'][slot.ti:slot.tf] = slot.type.amplitude.samples
        dictionary['det'][slot.ti:slot.tf] = slot.type.detuning.samples
        dictionary['phase'][slot.ti:slot.tf] = slot.type.phase
        return

    def _extract_samples(self):
        for channel in self._seq.declared_channels:
            # Extract globals
            if self._seq.declared_channels[channel].addressing == 'Global':

                self.addressing['global'] = True
                self._global_samples = self._make_sample_dict()

                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        self._record_samples_from_slot(slot, self._global_samples)

            # Extract Locals
            if self._seq.declared_channels[channel].addressing == 'Local':

                # Extract Local Rydberg schedules:
                if self._seq.declared_channels[channel].basis == 'ground-rydberg':

                    self.addressing['local_ryd'] = True
                    #self._local_ryd_samples = self._make_sample_dict()

                    for slot in self._seq._schedule[channel]:
                        if isinstance(slot.type, Pulse):
                            # for qubit in slot.targets:
                            # Get the target from set object
                            qubit = list(slot.targets)[0]
                            self._record_samples_from_slot(slot, self._local_ryd_samples[qubit])
                            # We assume that there are no two channels acting at the same time on the same qubit.

                # Extract Raman Local:
                if self._seq.declared_channels[channel].basis == 'digital':

                    self.addressing['digital'] = True
                    #self._local_raman_samples = self._make_sample_dict()

                    for slot in self._seq._schedule[channel]:
                        if isinstance(slot.type, Pulse):
                            # Get the target from set object
                            qubit = list(slot.targets)[0]
                            self._record_samples_from_slot(slot, self._local_raman_samples[qubit])

    def _create_basis(self,basis):
        """Creates the basis elements"""
        if basis == 'all':
            self._basis = {'r': qutip.basis(3, 0),
                           'g': qutip.basis(3, 1),
                           'h': qutip.basis(3, 2)
                           }
        if basis == 'ryd':
            self._basis = {'r': qutip.basis(2, 0),
                           'g': qutip.basis(2, 1)
                           }
        if basis == 'digital':
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

        if basis == 'ryd':
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
                               'sigma_gh': g * h.dag(),
                               'sigma_hh': h * h.dag(),
                               'sigma_gg': g * g.dag()
                               }

    # Define Operators
    def build_local_operator(self, operator, *qubit_ids):
        """
        Returns a local gate at atoms in positions given by qubit_ids
        """
        if len(set(qubit_ids)) < len(qubit_ids):
            raise ValueError("Duplicate atom ids in argument list")

        temp = [qutip.qeye(operator.shape[0]) for _ in range(self._L)] #List of identities with shape of operator
        for qubit_id in qubit_ids:
            pos = list(self._reg.qubits).index(qubit_id)
            temp[pos] = operator
        return qutip.tensor(temp)

    def _sum_local_operator(self, operator):
        """
        Returns a QObj for the sum of 'operator' over the entire array
        """
        ans = 0
        idd = [qutip.qeye(operator.shape[0]) for _ in range(self._L)]
        for pos in range(self._L):
            temp = idd.copy()
            temp[pos] = operator
            ans += qutip.tensor(temp)
        return ans

    def _make_VdW_term(self):
        """
        Constructs the Van der Waals Interaction Terms
        """
        VdW = 0
        # Get every pair without duplicates
        for qubit1, qubit2 in itertools.combinations(self._reg._ids, r=2):
            R = np.sqrt(
                np.sum((self._reg.qubits[qubit1] - self._reg.qubits[qubit2])**2))

            VdW += (1e5 / R**6) * 0.5 * self.build_local_operator(
                self._operators['sigma_rr'], qubit1, qubit2)
        return VdW

    def _make_rotation_term(self, addressing, qubit=None, local_samples=None, global_samples=None):
        """
        Creates the terms that go into the Hamiltonian. Note that this works
        once a basis has been defined.
        The (optional) global rydberg coefficients are added at the same place as the local rydberg
        """

        if addressing == 'rydberg':

            # Coefficient arrays:
            if global_samples:
                global_gr_coeff, global_n_coeff = global_samples

                # Build global Rydberg operators
                global_gr = self._sum_local_operator(self._operators['sigma_gr'])
                global_rr = self._sum_local_operator(self._operators['sigma_rr'])
            else:
                global_gr_coeff, global_n_coeff = 0, 0

            if local_samples:
                gr_coeff = local_samples['amp'] * np.exp(-1j * local_samples['phase'])
                n_coeff = local_samples['det']
            else:
                gr_coeff, n_coeff = 0, 0

            rotation_term = qutip.QobjEvo(
                [
                    # Include Global Channel and Local Rydberg Channels:
                    [self.build_local_operator(
                        self._operators['sigma_gr'], qubit), global_gr_coeff+gr_coeff],
                    [0.5 * self.build_local_operator(
                        self._operators['sigma_rr'], qubit), global_n_coeff+n_coeff]
                ],
                tlist=self._times)

        if addressing == 'digital':
            # Update local channel Coefficient arrays:
            hg_coeff = local_samples['amp'] * np.exp(-1j * local_samples['phase'])
            n_coeff = local_samples['det']

            # Digital basis
            rotation_term = qutip.QobjEvo(
                [
                    [self.build_local_operator(
                        self._operators['sigma_hg'], qubit), hg_coeff],
                    [0.5 * self.build_local_operator(
                        self._operators['sigma_hh'], qubit), n_coeff]
                ],
                tlist=self._times)

        return rotation_term


    def _construct_hamiltonian(self):
        # Check basis:
        if self.addressing['digital']:

            if self.addressing['global']+self.addressing['local_ryd'] == 0: # No use of rydberg: Use only digital

                self._create_basis('digital')
                self._create_operators('digital')

                self._H = 0
                for qubit in self._reg.qubits:
                    self._H += self._make_rotation_term('digital',
                                                        qubit,
                                                        local_samples = self._local_raman_samples[qubit]
                                                        )
            else: #Use three levels:

                self._create_basis('all')
                self._create_operators('all')

                # Van der Waals Interaction Terms
                self._H = qutip.QobjEvo( [self._make_VdW_term()] ) # Time independent

                if self.addressing['global']:
                    # Calculate once global channel coefficient arrays
                    global_gr_coeff = self._global_samples['amp'] * np.exp(-1j * self._global_samples['phase'])
                    global_n_coeff = self._global_samples['det']

                    if self.addressing['local_ryd']:   #Rydberg Global + Rydberg Local + Digital
                        # Add Local terms taken from 'local' channel
                        for qubit in self._reg.qubits:
                            self._H += self._make_rotation_term('rydberg',
                                                                qubit,
                                                                local_samples = self._local_ryd_samples[qubit],
                                                                global_samples = [global_gr_coeff,global_n_coeff]
                                                                )
                            self._H += self._make_rotation_term('digital',
                                                                qubit,
                                                                local_samples = self._local_raman_samples[qubit],
                                                                )
                    else: #Rydberg Global + Digital
                        for qubit in self._reg.qubits:
                            self._H += self._make_rotation_term('rydberg',
                                                                qubit,
                                                                global_samples = [global_gr_coeff,global_n_coeff]
                                                                )
                            self._H += self._make_rotation_term('digital',
                                                                qubit,
                                                                local_samples = self._local_raman_samples[qubit],
                                                                )
                else: #Rydberg Local + Digital
                    self._H += self._make_rotation_term('rydberg',
                                                        qubit,
                                                        local_samples = self._local_ryd_samples[qubit],
                                                        )
                    self._H += self._make_rotation_term('digital',
                                                        qubit,
                                                        local_samples = self._local_raman_samples[qubit],
                                                        )

        else: # Use only two-level Rydberg basis:

            self._create_basis('ryd')
            self._create_operators('ryd')

            self._H = qutip.QobjEvo( [self._make_VdW_term()] )

            if not self.addressing['global']: #Only Local
                for qubit in self._reg.qubits:
                    self._H += self._make_rotation_term('rydberg',
                                                        qubit,
                                                        local_samples = self._local_ryd_samples[qubit]
                                                        )

            elif not self.addressing['local']: #Only global
                global_gr_coeff = self._global_samples['amp'] * np.exp(-1j * self._global_samples['phase'])
                global_n_coeff = self._global_samples['det']

                self._H = qutip.QobjEvo( [self._make_VdW_term()] )
                self._H += self._make_rotation_term('rydberg',
                                                        global_samples = [global_gr_coeff,global_n_coeff]
                                                        )

            else: # Global+Local Rydberg
                # Calculate once global channel coefficient arrays
                global_gr_coeff = self._global_samples['amp'] * np.exp(-1j * self._global_samples['phase'])
                global_n_coeff = self._global_samples['det']

                self._H = qutip.QobjEvo( [self._make_VdW_term()] )
                for qubit in self._reg.qubits:
                    self._H += self._make_rotation_term('rydberg',
                                                        qubit,
                                                        local_samples = self._local_ryd_samples[qubit],
                                                        global_samples = [global_gr_coeff,global_n_coeff]
                                                        )
        self._H = self._H + (self._H).dag()

    # Run Simulation Evolution using Qutip
    def run(self, observable, plot=True):
        """
        Simulates the sequence with the predefined observable
        """
        result = qutip.sesolve(self._H, self._initial_state, self._times, [
                               observable])  # Without observables, we get the output state
        self.data = result.expect[0]

        if plot:
            plt.plot(self._times, self.data, label=f"Observable")
            plt.legend()
