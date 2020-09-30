import pulses

class Sequence():
    """
    Creates a pulse sequence
    """
    def __init__(self, device, atom_array):
        self.device = device
        self.queue = []
        # Check the atom_array is valid according to device
        print("STARTED SEQUENCE")

    def add(self,Pulse):
        self.queue.append(Pulse.rabi)
        # Check if burst is OK
        #print(f'ADDED BURST. SEQUENCE IS: {self.queue} ')

class CustomSequence(Sequence):
    """
    Creates a custom pulse sequence
    """
    def __init__(self, device, atom_array, initial_state):
        Sequence.__init__(self, device, atom_array, initial_state)

    # Additional methods

class ParamSequence(Sequence):
    """
    Creates a QAOA pulse sequence
    """
    def __init__(self, device, Parameters, initial_state='ALL_DOWN', layers=1):
        Sequence.__init__(self, device, initial_state)

        #Add first layer
        qaoa_pulse = pulses.IsingPulse(Parameters)
        self.add(qaoa_pulse)
        #Add the rest of layers
        for i in range(1,layers):
            qaoa_pulse(Parameters.rabi[i], Parameters.detunin[i])
            self.add(qaoa_pulse)
        print(f"BEGIN QAOA. INITIAL STATE IS {initial_state}")
