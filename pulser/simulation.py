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
    def __init__(self, sequence, register):
        self._seq = sequence
        self._reg = register
        self._L = len(self._reg.qubits)
        self._tot_duration = max([self._seq._last(ch).tf for ch in self._seq._schedule])
        self._times = np.arange(self._tot_duration, dtype=np.double)

        self.create_basis()
        self.create_operators()

        self.define_sample_dicts()
        self.extract_samples()

        self.construct_hamiltonian()

        self._initial_state = qutip.tensor([qutip.basis(3,0) for _ in range(self._L)])

    def define_sample_dicts(self):
        self._global_samples = {'amp':np.zeros(self._tot_duration),
                                'det':np.zeros(self._tot_duration),
                                'phase':np.zeros(self._tot_duration)}

        self._local_samples = {}
        for qubit in self._reg.qubits:
            self._local_samples[qubit] = {'amp':np.zeros(self._tot_duration),
                                          'det':np.zeros(self._tot_duration),
                                          'phase':np.zeros(self._tot_duration)}

    def extract_samples(self):
    #Extract globals

        for slot in self._seq._schedule['global']:
            if isinstance(slot.type,Pulse):
                self._global_samples['amp'][slot.ti:slot.tf] = slot.type.amplitude.samples
                self._global_samples['det'][slot.ti:slot.tf] = slot.type.detuning.samples
                self._global_samples['phase'][slot.ti:slot.tf] = slot.type.phase

        # Extract Pulse samples for local channel

        for slot in self._seq._schedule['local']:
            if isinstance(slot.type,Pulse):
                qubit = list(slot.targets)[0] # Get the target from set object
                self._local_samples[qubit]['amp'][slot.ti:slot.tf] = slot.type.amplitude.samples
                self._local_samples[qubit]['det'][slot.ti:slot.tf] = slot.type.detuning.samples
                self._local_samples[qubit]['phase'][slot.ti:slot.tf] = slot.type.phase


    def create_basis(self):
        """Creates the basis elements"""
        self._basis = {'r':qutip.basis(3,0),
                       'g':qutip.basis(3,1),
                       'h':qutip.basis(3,2)}

    def create_operators(self):
        r = self._basis['r']
        g = self._basis['g']
        h = self._basis['h']
        self._operators = { 'I' : qutip.qeye(3),

                            'sigma_gr' : r*g.dag() + g*r.dag(),
                            'rydY' : -1j*r*g.dag() + 1j*g*r.dag(),
                            'rydZ' : r*r.dag() - g*g.dag(),

                            'sigma_hg' : h*g.dag() + g*h.dag(),
                            'excY' : 1j*h*g.dag() - 1j*g*h.dag(),
                            'excZ' : -h*h.dag() + g*g.dag(),

                            'sigma_rr' : r*r.dag(),
                            'sigma_gg' : g*g.dag(),
                            'sigma_hh' : h*h.dag()
                           }

    #Define Operators

    def _local_operator(self,qubit_id,operator):
        """
        Returns a local gate at a qubit
        """
        temp = [qutip.qeye(3) for _ in range(self._L)]
        pos =  list(self._reg.qubits).index(qubit_id)
        temp[pos] = operator
        return qutip.tensor(temp)

    def _two_body_operator(self,qubit_id1, qubit_id2, operator):
        """
        Returns a local operator acting on specific positions
        Parameters:
            qubit_id1, qubit_id2: keys of the atoms in which the operator acts
            operator: Qobj for the operator
        """
        if qubit_id1 == qubit_id2:
            raise ValueError("Same Atom ID given for a Two-body operator")

        temp = [qutip.qeye(3) for _ in range(self._L)]
        pos1 =  list(self._reg.qubits).index(qubit_id1)
        pos2 =  list(self._reg.qubits).index(qubit_id2)
        temp[pos1] = operator
        temp[pos2] = operator

        return qutip.tensor(temp)

    def construct_hamiltonian(self):
        #Components for the Hamiltonian:

        #Van der Waals Interaction Terms
        VdW = 0
        for qubit1,qubit2 in itertools.combinations(self._reg._ids,r=2):
            R = np.sqrt(np.sum((self._reg.qubits[qubit1]-self._reg.qubits[qubit2])**2 ) )
            VdW += (1e5/R**6)*self._two_body_operator(qubit1,qubit2,self._operators['sigma_rr'])

        #Rydberg(Global) terms
        global_X = 0
        global_Y = 0
        global_N = 0
        for qubit in self._reg.qubits:
            global_X += self._local_operator(qubit,self._operators['sigma_gr']) # Rotation in the Ground-Rydberg basis
            global_Y += self._local_operator(qubit,self._operators['rydY'])
            global_N += self._local_operator(qubit,self._operators['sigma_rr'])


        #Build Hamiltonian

        # Build Hamiltonian as QobjEvo, using the register's coordinates
        self._H = qutip.QobjEvo( [VdW] ) # Time independent

        # Add Global X,Y and N terms with coefficients taken from 'global' channel
        self._H += qutip.QobjEvo( [global_N, self._global_samples['det']] , tlist=self._times  )
        self._H += qutip.QobjEvo( [global_X, self._global_samples['amp']*np.cos(self._global_samples['phase'])] , tlist=self._times  )
        self._H += qutip.QobjEvo( [global_Y, -self._global_samples['amp']*np.sin(self._global_samples['phase'])] , tlist=self._times  )

        # Add Local terms taken from 'local' channel
        for qubit in self._reg.qubits:
            # Update Coefficient lists:
            amplitude = self._local_samples[qubit]['amp']
            detuning = self._local_samples[qubit]['det']
            phase = self._local_samples[qubit]['phase']
            # Add local terms to Hamiltonian
            self._H+=qutip.QobjEvo([[self._local_operator(qubit, self._operators['sigma_gr']), amplitude*np.cos(phase)],
                              [self._local_operator(qubit, self._operators['rydY']), -amplitude*np.sin(phase)],
                              [self._local_operator(qubit, self._operators['rydZ']), detuning]],
                             tlist=self._times)
            self._H+=qutip.QobjEvo([[self._local_operator(qubit, self._operators['sigma_hg']), amplitude*np.cos(phase)],
                              [self._local_operator(qubit, self._operators['excY']), -amplitude*np.sin(phase)],
                              [self._local_operator(qubit, self._operators['excZ']), detuning]],
                             tlist=self._times)

    #Evolution using Qutip
    # Add magnetization (in Rydberg basis) as observable
    def simulate(self, operator, plot=True):
        """
        Simulates the same local operator in every atom, e.g. total magnetization
        """
        observable = 0
        for qubit in self._reg.qubits:
            observable += self._local_operator(qubit,operator)

        result = qutip.sesolve(self._H, self._initial_state, self._times, [observable]) # Without observables, we get the output state
        self.data = result.expect[0]

        if plot:
            plt.plot(self._times,self.data,label=f"Magnetization")
            plt.legend()
