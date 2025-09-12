# Copyright 2023 Pulser Development Team
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
"""Defines a series of physical constants.

These are environment variables that can be overwritten.
"""
from os import getenv

TRAP_WAVELENGTH = float(getenv("TRAP_WAVELENGTH", 0.85))  # µm
MASS = float(getenv("MASS", 1.45e-25))  # kg
KB = float(getenv("KB", 1.38e-23))  # J/K
KEFF = float(getenv("KEFF", 8.7))  # µm^-1
