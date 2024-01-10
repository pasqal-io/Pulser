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

# Redundant imports are necessary to avoid errors with pyright

from pulser._version import __version__ as __version__
from pulser.pulse import Pulse
from pulser.register import Register, Register3D
from pulser.sequence import Sequence

from pulser.backend import QPUBackend  # isort: skip

# Public submodules
__pulser_submodules__ = [
    "backend",
    "devices",
    "register",
    "sampler",
    "waveforms",
]

__all__ = __pulser_submodules__ + [
    "Pulse",
    "Register",
    "Register3D",
    "Sequence",
    "QPUBackend",
]
