from abc import ABC, abstractmethod

import numpy as np
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
        self.add_layers()
        print(f"BEGIN QAOA. INITIAL STATE IS {initial_state}")

    def add_layers(self):
        # Try a simple sequence of square pulses
        layer_params = self.get_layer_params(0)
        pulse = pulses.IsingPulse(layer_params)
        self.add(pulse)
        for i in range(1,self.depth):
            layer_params = self.get_layer_params(i)
            pulse(layer_params)
            self.add(pulse)

    def get_layer_params(self, pos):
        layer_params = {
                        "rabi" : self.parameters["rabi"][pos],
                        "detuning" : self.parameters["detuning"][pos],
                        "intervals" : self.parameters["intervals"][pos]
                        }
        return layer_params

    def get_parameter_set(self):
        ans = []
        for key,values in self.parameters.items():
            #print(f"{key}:")
            ans.append(values)
        return ans

    # There should be a function to modify the existing pulse
