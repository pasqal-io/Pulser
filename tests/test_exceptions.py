"""Trivial tests on our exceptions."""

import pytest


def test_import_old_serialize_exceptions():
    """Test importing from pulser.json.exceptions.

    This should trigger a warning.
    """
    with pytest.warns(
        expected_warning=DeprecationWarning,
        match="module pulser.json.exceptions is deprecated",
    ):
        from pulser.json.exceptions import SerializationError

        _ = SerializationError
