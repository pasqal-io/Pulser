
class Pulse():
    """
    Base pulse class. For the moment onyl global pulses?
    """
    def __init__(self, parameters):
        self.rabi = parameters["rabi"]
        self.detuning = parameters["detuning"]
        self.intervals = parameters["intervals"]

    # add abstract methods for pulses?

class IsingPulse(Pulse):
    """
    Creates Ising Pulse
    """
    def __init__(self, parameters):
        super().__init__(parameters)
        print("LOADED PULSE")

    def __call__(self, parameters):
        if parameters["rabi"] is not None: self.rabi = parameters["rabi"]
        if parameters["detuning"] is not None: self.detuning = parameters["detuning"]
        if parameters["intervals"] is not None: self.intervals = parameters["intervals"]
        print("UPDATED PULSE")
