from .operations import *
from .devices import *

def program():
    """
    Creates a program object that can be either simulated or run through a PasQal device
    """
    def __init__(self, pulse_seq, parameters, classical_process='optimize'):
        self.pulse_seq = pulse_seq
        self.parameters = parameters
        self.classical_process = classical_process
