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
"""Interaction coefficients for Rydberg levels between 50 and 100.

Stored values and units:
- C_6/hbar: rad/µs x µm^6
- C_3/hbar: rad/µs x µm^3

The values were calculated using ARC_ and double checked with
PairInteraction_.

.. _ARC: https://arc-alkali-rydberg-calculator.readthedocs.io/
.. _PairInteraction: https://www.pairinteraction.org/
"""

import json
from pathlib import PurePath

with open(
    PurePath(__file__).parent / "C6_coeffs.json", "r", encoding="utf-8"
) as f:
    _json_dict = json.load(f)
c6_dict = {int(key): value for key, value in _json_dict.items()}

with open(
    PurePath(__file__).parent / "C3_coeffs.json", "r", encoding="utf-8"
) as f:
    _json_dict = json.load(f)
c3_dict = {int(key): value for key, value in _json_dict.items()}
