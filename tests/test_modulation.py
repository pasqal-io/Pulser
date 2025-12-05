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
"""Tests for the modulation bandwidth utilities."""

import warnings

import numpy as np
import pytest

from pulser.channels.modulation import (
    calculate_amplitude_rise_time,
    calculate_mod_bandwidth_from_amplitude_rise_time,
    calculate_mod_bandwidth_from_intensity_rise_time,
    validate_mod_bandwidth,
)


class TestModulationConversions:
    """Tests for modulation bandwidth and rise time conversions."""

    def test_intensity_rise_time_roundtrip(self):
        """Test that intensity rise time conversions are consistent."""
        # Known value: MODBW_TO_TR = 0.48, so for mod_bw=1 MHz:
        # intensity_rise_time = 0.48 / 1 * 1e3 = 480 ns
        mod_bw = calculate_mod_bandwidth_from_intensity_rise_time(480)
        assert np.isclose(mod_bw, 1.0, rtol=1e-3)

    def test_amplitude_rise_time_calculation(self):
        """Test amplitude rise time calculation from modulation bandwidth."""
        # For mod_bw = 1 MHz:
        # intensity_rise_time = 480 ns
        # amplitude_rise_time = 480 * sqrt(2) ≈ 679 ns
        rise_time = calculate_amplitude_rise_time(1.0)
        expected = int(round(480 * np.sqrt(2)))
        assert rise_time == expected

    def test_amplitude_to_mod_bandwidth_roundtrip(self):
        """Test roundtrip conversion between amplitude rise time and mod_bw."""
        original_rise_time = 100  # ns
        mod_bw = calculate_mod_bandwidth_from_amplitude_rise_time(
            original_rise_time
        )
        recovered_rise_time = calculate_amplitude_rise_time(mod_bw)
        # Allow for rounding error (result is int)
        assert abs(recovered_rise_time - original_rise_time) <= 1

    def test_intensity_vs_amplitude_rise_time_relationship(self):
        """Test that amp rise time is sqrt(2) times intensity rise time."""
        mod_bw = 5.0  # MHz
        amp_rise_time = calculate_amplitude_rise_time(mod_bw)
        # intensity_rise_time = 0.48 / 5 * 1e3 = 96 ns
        int_rise_time = 0.48 / mod_bw * 1e3
        expected_amp_rise_time = int(round(int_rise_time * np.sqrt(2)))
        assert amp_rise_time == expected_amp_rise_time


class TestValidateModBandwidth:
    """Tests for modulation bandwidth validation."""

    def test_valid_mod_bandwidth(self):
        """Test that valid mod_bandwidth values pass validation."""
        # Should not raise
        validate_mod_bandwidth(1.0)
        validate_mod_bandwidth(100.0)
        validate_mod_bandwidth(0.001)

    def test_zero_mod_bandwidth_raises(self):
        """Test that zero mod_bandwidth raises ValueError."""
        with pytest.raises(
            ValueError, match="'mod_bandwidth' must be greater than zero"
        ):
            validate_mod_bandwidth(0.0)

    def test_negative_mod_bandwidth_raises(self):
        """Test that negative mod_bandwidth raises ValueError."""
        with pytest.raises(
            ValueError, match="'mod_bandwidth' must be greater than zero"
        ):
            validate_mod_bandwidth(-5.0)

    def test_excessive_mod_bandwidth_raises(self):
        """Test that mod_bandwidth above max raises NotImplementedError."""
        max_bw = calculate_mod_bandwidth_from_amplitude_rise_time(1)
        with pytest.raises(
            NotImplementedError,
            match=f"'mod_bandwidth' must be lower than {max_bw:.0f} MHz",
        ):
            validate_mod_bandwidth(max_bw + 1)


class TestDeprecationWarnings:
    """Tests for MODBW_TO_TR deprecation warnings."""

    def test_modbw_to_tr_deprecation_from_eom(self):
        """Test that importing MODBW_TO_TR from eom raises warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from pulser.channels.eom import MODBW_TO_TR

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "MODBW_TO_TR" in str(w[0].message)
            assert "pulser.channels.eom" in str(w[0].message)
            # Value should still work for backward compatibility
            assert MODBW_TO_TR == 0.48

    def test_modbw_to_tr_deprecation_from_base_channel(self):
        """Test that importing MODBW_TO_TR from base_channel raises warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from pulser.channels.base_channel import MODBW_TO_TR

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "MODBW_TO_TR" in str(w[0].message)
            assert "pulser.channels.base_channel" in str(w[0].message)
            # Value should still work for backward compatibility
            assert MODBW_TO_TR == 0.48
