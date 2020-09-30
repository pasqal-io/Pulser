class Pulse():
    """
    Base pulse class. For the moment onyl global pulses?
    """
    def __init__(self, Parameters):
        self.rabi = Parameters.rabi
        self.detuning = Parameters.detuning
        self.interval = Parameters.interval
        print("LOADED")
        # Check this is a good burst

class IsingPulse(Pulse):
    """
    Creates Ising Pulse
    """
    def __init__(self, Parameters):
        Pulse.__init__(self, Parameters)

    def __call__(self, Parameters):
        if Parameters.rabi is not None: self.rabi = Parameters.rabi
        if Parameters.detuning is not None: self.detuning = Parameters.detuning
        if Parameters.interval is not None: self.interval = Parameters.interval
        print("UPDATED")

class Parameters():
    def __init__(self, rabi, detuning, interval):
        self.rabi = rabi
        self.detuning = detuning
        self.interval = interval
