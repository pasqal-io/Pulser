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
"""C_6/hbar (in  um^6 / us`), coeffs for Rydberg levels between 50 and 100."""

import json
from pathlib import PurePath

_json_dict = json.load(open(PurePath(__file__).parent / "C6_coeffs.json"))
c6_dict = {int(key): value for key, value in _json_dict.items()}
