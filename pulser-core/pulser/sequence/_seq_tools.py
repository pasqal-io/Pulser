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
"""Some tools to be used in Sequence."""


def dmm_id_from_name(dmm_name: str) -> str:
    """Converts a dmm_name into a dmm_id.

    As a reminder the dmm_name is generated automatically from dmm_id
    as dmm_id_{number of times dmm_id has been called}.

    Args:
        dmm_name: The dmm_name to convert.
    Returns:
        The associated dmm_id.
    """
    return "_".join(dmm_name.split("_")[0:2])


def get_dmm_name(dmm_id: str, channels: list[str]) -> str:
    """Get the dmm_name to add a dmm_id to a list of channels.

    Counts the number of channels starting by dmm_id, generates the
    dmm_name as dmm_id_{number of times dmm_id has been called}.

    Args:
        dmm_id: the id of the DMM to add to the list of channels.
        channels: a list of channel names.
    Returns:
        The associated dmm_name.
    """
    dmm_count = len(
        [key for key in channels if dmm_id_from_name(key) == dmm_id]
    )
    if dmm_count == 0:
        return dmm_id
    return dmm_id + f"_{dmm_count}"
