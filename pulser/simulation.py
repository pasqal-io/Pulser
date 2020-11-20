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

        self.create_basis()
        self.create_operators()

        self.define_sample_dicts()
        self.extract_samples()

        self.construct_hamiltonian()

        self._initial_state = qutip.tensor(
            [qutip.basis(3, 0) for _ in range(self._L)])

    def define_sample_dicts(self):
        self._global_samples = {'amp': np.zeros(self._tot_duration),
                                'det': np.zeros(self._tot_duration),
                                'phase': np.zeros(self._tot_duration)}

        self._local_ryd_samples = {}
        for qubit in self._reg.qubits:
            self._local_ryd_samples[qubit] = {'amp': np.zeros(self._tot_duration),
                                              'det': np.zeros(self._tot_duration),
                                              'phase': np.zeros(self._tot_duration)}
        self._local_raman_samples = {}
        for qubit in self._reg.qubits:
            self._local_raman_samples[qubit] = {'amp': np.zeros(self._tot_duration),
                                                'det': np.zeros(self._tot_duration),
                                                'phase': np.zeros(self._tot_duration)}

    def extract_samples(self):
        for channel in self._seq.declared_channels:
            # Extract globals
            if self._seq.declared_channels[channel].addressing == 'Global':

                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        self._global_samples['amp'][slot.ti:
                                                    slot.tf] = slot.type.amplitude.samples
                        self._global_samples['det'][slot.ti:
                                                    slot.tf] = slot.type.detuning.samples
                        self._global_samples['phase'][slot.ti:slot.tf] = slot.type.phase

            # Extract Locals
            if self._seq.declared_channels[channel].addressing == 'Local':
                # Extract Local Rydberg schedules:
                if self._seq.declared_channels[channel].basis == 'ground-rydberg':
                    for slot in self._seq._schedule[channel]:
                        if isinstance(slot.type, Pulse):

                            # for qubit in slot.targets:
                            # Get the target from set object
                            qubit = list(slot.targets)[0]
                            # We assume that there are no two channels acting at the same time on the same qubit.
                            # This implies that one can add the samples in the current interval
                            # and not worry if the other local channel is acting also at those times.
                            # But, are the two local rydberg channels necessary identical?
                            self._local_ryd_samples[qubit]['amp'][slot.ti:
                                                                  slot.tf] = slot.type.amplitude.samples
                            self._local_ryd_samples[qubit]['det'][slot.ti:
                                                                  slot.tf] = slot.type.detuning.samples
                            self._local_ryd_samples[qubit]['phase'][slot.ti:slot.tf] = slot.type.phase
                # Extract Raman Local:
                if self._seq.declared_channels[channel].basis == 'digital':
                    for slot in self._seq._schedule[channel]:
                        if isinstance(slot.type, Pulse):
                            # Get the target from set object
                            qubit = list(slot.targets)[0]
                            self._local_raman_samples[qubit]['amp'][slot.ti:
                                                                    slot.tf] = slot.type.amplitude.samples
                            self._local_raman_samples[qubit]['det'][slot.ti:
                                                                    slot.tf] = slot.type.detuning.samples
                            self._local_raman_samples[qubit]['phase'][slot.ti:slot.tf] = slot.type.phase

    def create_basis(self):
        """Creates the basis elements"""
        self._basis = {'r': qutip.basis(3, 0),
                       'g': qutip.basis(3, 1),
                       'h': qutip.basis(3, 2)}

    def create_operators(self):
        r = self._basis['r']
        g = self._basis['g']
        h = self._basis['h']
        self._operators = {'I': qutip.qeye(3),
                           'sigma_gr': g * r.dag(),
                           'sigma_rg': g.dag() * r,

                           'sigma_hg': h * g.dag(),
                           'sigma_gh': h.dag() * g,

                           'sigma_rr': r * r.dag(),
                           'sigma_gg': g * g.dag(),
                           'sigma_hh': h * h.dag()
                           }

    # Define Operators

    def build_local_operator(self, operator, *qubit_ids):
        """
        Returns a local gate at atoms in positions given by qubit_ids
        """
        if len(set(qubit_ids)) < len(qubit_ids):
            raise ValueError("Duplicate atom ids in argument list")

        temp = [qutip.qeye(3) for _ in range(self._L)]
        for qubit_id in qubit_ids:
            pos = list(self._reg.qubits).index(qubit_id)
            temp[pos] = operator
        return qutip.tensor(temp)

    def construct_hamiltonian(self):
        # Components for the Hamiltonian:

        # Van der Waals Interaction Terms
        VdW = 0
        # Get every pair without duplicates
        for qubit1, qubit2 in itertools.combinations(self._reg._ids, r=2):
            R = np.sqrt(
                np.sum((self._reg.qubits[qubit1] - self._reg.qubits[qubit2])**2))
            VdW += (1e5 / R**6) * \
                self.build_local_operator(
                    self._operators['sigma_rr'], qubit1, qubit2)

        # Rydberg(Global) terms
        global_gr = 0
        global_rr = 0
        for qubit in self._reg.qubits:
            # Rotation in the Ground-Rydberg basis
            global_gr += self.build_local_operator(
                self._operators['sigma_gr'], qubit)

            global_rg += self.build_local_operator(
                self._operators['sigma_gr'], qubit)

            global_rr += self.build_local_operator(
                self._operators['sigma_rr'], qubit)

        # Build Hamiltonian

        # Build Hamiltonian as QobjEvo, using the register's coordinates
        self._H = qutip.QobjEvo([VdW])  # Time independent

        # Calculate once global channel coefficient arrays
        global_gr_coeff = self._global_samples['amp'] * \
            np.exp(-1j * self._global_samples['phase'])

        global_rg_coeff = self._global_samples['amp'] * \
            np.exp(1j * self._global_samples['phase'])

        global_n_coeff = self._global_samples['det']

        # Add Local terms taken from 'local' channel
        for qubit in self._reg.qubits:
            # Update local channel Coefficient arrays:
            amplitude = self._local_ryd_samples[qubit]['amp']
            detuning = self._local_ryd_samples[qubit]['det']
            phase = self._local_ryd_samples[qubit]['phase']

            # Add local terms to Hamiltonian
            # Ground-Rydberg basis
            self._H += qutip.QobjEvo(
                [
                    # Include Global Channel and Local Rydberg Channels:
                    [self.build_local_operator(
                        self._operators['sigma_gr'], qubit), global_gr_coeff + amplitude * np.exp(-1j * phase)],
                    [self.build_local_operator(
                        self._operators['sigma_rg'], qubit), global_rg_coeff + amplitude * np.exp(1j * phase)],
                    [self.build_local_operator(
                        self._operators['sigma_rr'], qubit), global_n_coeff + detuning]
                ],
                tlist=self._times)

            # Update local channel Coefficient arrays:
            amplitude = self._local_raman_samples[qubit]['amp']
            detuning = self._local_raman_samples[qubit]['det']
            phase = self._local_raman_samples[qubit]['phase']
            # Digital basis
            self._H += qutip.QobjEvo(
                [
                    [self.build_local_operator(
                        self._operators['sigma_hg'], qubit), amplitude * np.exp(-1j * phase)],
                    [self.build_local_operator(
                        self._operators['sigma_gh'], qubit), amplitude * np.exp(1j * phase)],
                    [self.build_local_operator(
                        self._operators['sigma_hh'], qubit), detuning]
                ],
                tlist=self._times)

    # Run Simulation Evolution using Qutip
    def run(self, observable, plot=True):
        """
        Simulates the sequence with the predefined observable
        """
        result = qutip.sesolve(self._H, self._initial_state, self._times, [
                               observable])  # Without observables, we get the output state
        self.data = result.expect[0]

        if plot:
            plt.plot(self._times, self.data, label=f"{str(observable)}")
            plt.legend()
