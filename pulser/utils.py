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

import warnings


def validate_duration(duration, min_duration=16, max_duration=67108864):
    """Validates a time interval.

    Returns:
        int: The duration in multiples of 4 ns.
    """
    try:
        _duration = int(duration)
    except (TypeError, ValueError):
        raise TypeError("duration needs to be castable to an int but "
                        "type %s was provided" % type(duration))

    if duration < min_duration:
        raise ValueError(f"duration has to be at least {min_duration} ns.")

    if duration > max_duration:
        raise ValueError(f"duration can be at most {max_duration} ns.")

    if duration % 4 != 0:
        _duration -= _duration % 4
        warnings.warn("The given duration is below the machine's precision"
                      " of 4 ns time steps. It was rounded down to the"
                      " nearest multiple of 4 ns.")
    return _duration
