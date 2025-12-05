# Copyright 2025 Pulser Development Team
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
"""Utilities for modulation bandwidth and rise time calculations."""

import numpy as np

__all__ = [
    "calculate_mod_bandwidth_from_intensity_rise_time",
    "calculate_mod_bandwidth_from_amplitude_rise_time",
    "calculate_amplitude_rise_time",
    "validate_mod_bandwidth",
]


def _mod_bw_rise_time_conversion(input_value: float) -> float:
    """Converts between modulation bandwidth and intensity rise time.

    This is a bidirectional conversion function that uses the empirically
    derived conversion factor.

    Args:
        input_value: Either modulation bandwidth (in MHz) or intensity rise
            time (in ns).

    Returns:
        If input is mod_bandwidth, returns intensity_rise_time (in ns).
        If input is intensity_rise_time, returns mod_bandwidth (in MHz).
    """
    # Conversion factor from modulation bandwidth to rise time
    MODBW_TO_TR = 0.48
    return MODBW_TO_TR / input_value * 1e3


def calculate_mod_bandwidth_from_intensity_rise_time(
    intensity_rise_time: int,
) -> float:
    """Calculate the modulation bandwidth from the intensity rise time.

    Warning:
        For legacy reasons, the modulation bandwidth used in Pulser
        follows an unorthodox definition: it corresponds to the
        frequency component that experiences a 75% attenuation in
        amplitude.
        It corresponds to 2x the standard modulation bandwidth (defined
        as the frequency at which there is a 50% attenuation in power,
        i.e. 3dB attenuation).

    Args:
        intensity_rise_time: The time taken to go from 10% to 90% output
            power in response to a step change in the input (in ns).

    Returns:
        mod_bandwidth: The modulation bandwidth (in MHz), following
        Pulser's definition.
    """
    return _mod_bw_rise_time_conversion(intensity_rise_time)


def calculate_mod_bandwidth_from_amplitude_rise_time(
    amplitude_rise_time: int,
) -> float:
    """Calculate the modulation bandwidth from the amplitude rise time.

    Warning:
        For legacy reasons, the modulation bandwidth used in Pulser
        follows an unorthodox definition: it corresponds to the
        frequency component that experiences a 75% attenuation in
        amplitude.
        It corresponds to 2x the standard modulation bandwidth (defined
        as the frequency at which there is a 50% attenuation in power,
        i.e. 3dB attenuation).

    Args:
        amplitude_rise_time: The time taken to go from 10% to 90% output
            amplitude in response to a step change in the input (in ns).

    Returns:
        mod_bandwidth: The modulation bandwidth (in MHz), following
        Pulser's definition.
    """
    return calculate_mod_bandwidth_from_intensity_rise_time(
        amplitude_rise_time
        / np.sqrt(2)  # amp_rise_time = sqrt(2) * int_rise_time
    )


def calculate_amplitude_rise_time(mod_bandwidth: float) -> int:
    """Calculate the amplitude rise time from the modulation bandwidth.

    Not to be confused with the rise time in intensity, which is the
    value usually measured experimentally.

    Args:
        mod_bandwidth: The modulation bandwidth (in MHz).

    Returns:
        The amplitude rise time (in ns), defined as the time taken to go
        from 10% to 90% output amplitude in response to a step change.

    Note:
        The calculation is based on the modulation bandwidth at which
        there is a 75% attenuation in amplitude. The relationship between
        amplitude and intensity rise times is: t_amp = sqrt(2) * t_int.
    """
    return int(round(_mod_bw_rise_time_conversion(mod_bandwidth) * np.sqrt(2)))


def validate_mod_bandwidth(mod_bandwidth: float) -> None:
    """Validate that the modulation bandwidth is within acceptable limits.

    Args:
        mod_bandwidth: The modulation bandwidth (in MHz) to validate.

    Raises:
        ValueError: If mod_bandwidth is not greater than zero.
        NotImplementedError: If mod_bandwidth exceeds the maximum allowed
        value.
    """
    if mod_bandwidth <= 0.0:
        raise ValueError(
            "'mod_bandwidth' must be greater than zero, not"
            f" {mod_bandwidth}."
        )
    if mod_bandwidth > (
        max_bw := calculate_mod_bandwidth_from_amplitude_rise_time(1)
    ):
        raise NotImplementedError(
            f"'mod_bandwidth' must be lower than {max_bw:.0f} MHz"
        )
