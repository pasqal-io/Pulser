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
