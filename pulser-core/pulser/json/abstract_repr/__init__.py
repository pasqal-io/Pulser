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
"""Serialization and deserialization tools for the abstract representation."""
import json
from pathlib import Path

SCHEMAS_PATH = Path(__file__).parent / "schemas"
SCHEMAS = {}
for obj_type in (
    "device",
    "sequence",
    "register",
    "layout",
    "noise",
    "results",
    "config",
):
    with open(
        SCHEMAS_PATH / f"{obj_type}-schema.json", "r", encoding="utf-8"
    ) as f:
        SCHEMAS[obj_type] = json.load(f)
