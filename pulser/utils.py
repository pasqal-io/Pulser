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


def validate_duration(duration):
    """Validates a time interval.

    Returns:
        int: The duration in ns.
    """
    try:
        _duration = int(duration)
    except (TypeError, ValueError):
        raise TypeError("duration needs to be castable to an int but "
                        "type %s was provided" % type(duration))

    if duration <= 0:
        raise ValueError("duration has to be castable to a positive "
                         "integer.")

    if duration % 1 != 0:
        warnings.warn("The given duration is below the machine's precision"
                      " of 1 ns time steps. It was rounded down to the"
                      " nearest integer.")
    return _duration
