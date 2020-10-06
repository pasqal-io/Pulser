from abc import ABC, abstractmethod


class Channel(ABC):
    """Base class for an hardware channel."""

    def __init__(self, addressing, max_abs_detuning, max_amp):
        """Initialize a channel with specific charactheristics.

        Args:
            addressing (str): 'local' or 'global'.
            max_abs_detuning (tuple): Maximum possible detuning (in MHz), in
            absolute value.
            max_amp(tuple): Maximum pulse amplitude (in MHz).
        """
        if addressing not in ['global', 'local']:
            raise ValueError("Addressing can only be 'global' or 'local'.")
        self.addressing = addressing

        if max_abs_detuning < 0:
            raise ValueError("Maximum absolute detuning has to be positive.")
        self.max_abs_detuning = max_abs_detuning

        if max_amp <= 0:
            raise ValueError("Maximum channel amplitude has to be positive.")
        self.max_amp = max_amp

    @property
    @abstractmethod
    def name(self):
        pass

    def __repr__(self):
        s = "({}, Max Absolute Detuning: {} MHz, Max Amplitude: {} MHz)"
        config = s.format(self.addressing, self.max_abs_detuning, self.max_amp)
        return self.name + config


class Raman(Channel):
    """Raman beam channel."""
    @property
    def name(self):
        return 'Raman'


class Rydberg(Channel):
    """Rydberg beam  channel."""
    @property
    def name(self):
        return 'Rydberg'


class MW(Channel):
    """Microwave channel."""
    @property
    def name(self):
        return 'MW'
