from abc import ABC, abstractmethod
import pulses

class Sequence(ABC):
    """
    Creates a pulse sequence. A sequence is composed by
    - The device in which we want to implement this sequence
    - The parameters of each of the composing pulses
    - The current queue of pulses to which we can add , modify or delete
    """
    def __init__(self, device, parameters, initial_state):
        self.device = device
        self.parameters = parameters
        self.initial_state = initial_state
        self.queue = []
        # Maybe add an 'empty' pulse by default


    def add(self,Pulse):
        self.queue.append(Pulse)
        # Check if burst is OK
        #print(f'ADDED BURST. SEQUENCE IS: {self.queue} ')

    @abstractmethod
    def add_layers(self, Parameters):
        pass

class CustomSequence(Sequence):
    """
    Creates a custom pulse sequence
    """
    def __init__(self, device, atom_array, initial_state):
        super().__init__(device, initial_state)

    # Additional methods

class ParamSequence(Sequence):
    """
    Creates a predetermined pulse sequence for variational programs with several layers.
    Recall that parameters can hold information about the type of function that
    represents rabi or detuning, and whose defining parameters are being optimized
    """
    def __init__(self, device, param_set, initial_state='ALL_DOWN'):
        super().__init__(device, param_set, initial_state)
        self.depth = len(param_set["intervals"])
        print("SEQUENCE ACTIVATED")
        self.add_layers(param_set)
        print(f"BEGIN QAOA. INITIAL STATE IS {initial_state}")

    def add_layers(self, param_set):
        # Try a simple sequence of square pulses
        layer_params = self.get_layer_params(param_set, 0)
        pulse = pulses.IsingPulse(layer_params)
        self.add(pulse)
        for i in range(1,self.depth):
            layer_params = self.get_layer_params(param_set, i)
            pulse(layer_params)
            self.add(pulse)

    def get_layer_params(self, param_set, position):
        layer_params = {
                        "rabi" : param_set["rabi"][position],
                        "detuning" : param_set["detuning"][position],
                        "intervals" : param_set["intervals"][position]
                        }
        return layer_params

    # There should be a function to modify the existing pulse

if __name__ == "__main__":

    depth = 5

    # Define parameters:
    params = {
                "rabi" : [f"rabi{k}" for k in range(depth)],
                "detuning" : [f"detuning{k}" for k in range(depth)],
                "intervals" : [ 0.1*k for k in range(depth)]
                }
    # can add also identifiers for a particular type of function (square, sawtooth, etc).
    # the associated parameters should still remain as numbers

    seq = ParamSequence(device="my_device", param_set=params)
    print(seq.queue)
