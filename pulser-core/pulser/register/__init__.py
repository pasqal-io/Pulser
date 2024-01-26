# Copyright 2022 Pulser Development Team
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
"""Classes for qubit register definition."""

from pulser.register.base_register import QubitId
from pulser.register.register import Register
from pulser.register.register3d import Register3D
from pulser.register.register_layout import RegisterLayout
from pulser.register.special_layouts import (
    SquareLatticeLayout,
    TriangularLatticeLayout,
)

__all__ = [
    "QubitId",
    "Register",
    "Register3D",
    "RegisterLayout",
    "SquareLatticeLayout",
    "TriangularLatticeLayout",
]
