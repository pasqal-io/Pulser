import bursts

class Sequence():
    """
    Creates a pulse sequence
    """
    def __init__(self, device, atom_array, initial_state):
        self.device = device
        self.atom_array = atom_array
        self.initial_state = initial_state
        self.queue = []
        # Check the atom_array is valid according to device
        print("STARTED SEQUENCE")

    def add(self,Burst):
        self.queue.append(Burst.rabi)
        # Check if burst is OK
        #print(f'ADDED BURST. SEQUENCE IS: {self.queue} ')

class CustomSequence(Sequence):
    """
    Creates a custom pulse sequence
    """
    def __init__(self, device, atom_array, initial_state):
        Sequence.__init__(self, device, atom_array, initial_state)

    # Additional methods

class QaoaSequence(Sequence):
    """
    Creates a QAOA pulse sequence
    """
    def __init__(self, device, Parameters, initial_state, layers=1):
        Sequence.__init__(self, device, atom_array='GLOBAL', initial_state='ALL_DOWN')
        #Add first layer
        qaoa_burst = bursts.IsingBurst(Parameters)
        self.add(qaoa_burst)
        #Add the rest of layers
        for i in range(1,layers):
            Parameters.rabi=f"rabi_{i+1}"
            qaoa_burst(Parameters)
            self.add(qaoa_burst)
        print(f"BEGIN QAOA. INITIAL STATE IS {initial_state}")
