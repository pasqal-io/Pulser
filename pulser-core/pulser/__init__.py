# Copyright 2020 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A pulse-level composer for neutral-atom quantum devices."""

from pulser._version import __version__ as __version__
from pulser.waveforms import (
    CompositeWaveform,
    CustomWaveform,
    ConstantWaveform,
    RampWaveform,
    BlackmanWaveform,
    InterpolatedWaveform,
    KaiserWaveform,
)
from pulser.pulse import Pulse
from pulser.register import Register, Register3D
from pulser.noise_model import NoiseModel
from pulser.devices import AnalogDevice, DigitalAnalogDevice, MockDevice
from pulser.sequence import Sequence
from pulser.backend import (
    EmulatorConfig,
    QPUBackend,
)

# Exposing relevant submodules
from pulser import (
    waveforms as waveforms,
    channels as channels,
    register as register,
    devices as devices,
    sampler as sampler,
    backend as backend,
    backends as backends,
)

__all__ = [
    # pulser.waveforms
    "CompositeWaveform",
    "CustomWaveform",
    "ConstantWaveform",
    "RampWaveform",
    "BlackmanWaveform",
    "InterpolatedWaveform",
    "KaiserWaveform",
    # pulser.pulse
    "Pulse",
    # pulser.register
    "Register",
    "Register3D",
    # pulser.noise_model
    "NoiseModel",
    # pulser.devices
    "AnalogDevice",
    "DigitalAnalogDevice",
    "MockDevice",
    # pulser.sequence
    "Sequence",
    # pulser.backends
    "EmulatorConfig",
    "QPUBackend",
]
