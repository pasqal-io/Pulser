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
"""Custom decorators used by the Sequence class."""
from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from itertools import chain
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pulser.parametrized import Parametrized
from pulser.sequence._call import _Call

if TYPE_CHECKING:
    from pulser.sequence.sequence import Sequence

F = TypeVar("F", bound=Callable)


def screen(func: F) -> F:
    """Blocks the call to a function if the Sequence is parametrized."""

    @wraps(func)
    def wrapper(self: Sequence, *args: Any, **kwargs: Any) -> Any:
        if self.is_parametrized():
            raise RuntimeError(
                f"Sequence.{func.__name__} can't be called in"
                " parametrized sequences."
            )
        return func(self, *args, **kwargs)

    return cast(F, wrapper)


def verify_variable(seq: Sequence, x: Any) -> None:
    """Checks if a variable has been declared in a sequence."""
    if isinstance(x, Parametrized):
        # If not already, the sequence becomes parametrized
        seq._building = False
        for name, var in x.variables.items():
            if name not in seq._variables:
                raise ValueError(f"Unknown variable '{name}'.")
            elif seq._variables[name] is not var:
                raise ValueError(
                    f"{x} has variables that don't come from this "
                    "Sequence. Use only what's returned by this"
                    "Sequence's 'declare_variable' method as your"
                    "variables."
                )
    elif not isinstance(x, str):
        # Recursively look for parametrized objs inside the arguments
        try:
            for y in x:
                verify_variable(seq, y)
        except TypeError:
            # x is not iterable, do nothing
            pass


def verify_parametrization(func: F) -> F:
    """Checks and updates the sequence status' consistency with the call.

    - Checks the sequence can still be modified.
    - Checks if all Parametrized inputs stem from declared variables.
    """

    @wraps(func)
    def wrapper(self: Sequence, *args: Any, **kwargs: Any) -> Any:
        for x in chain(args, kwargs.values()):
            verify_variable(self, x)
        func(self, *args, **kwargs)

    return cast(F, wrapper)


def store(func: F) -> F:
    """Checks and stores the call to call it when building the Sequence."""

    @wraps(func)
    @verify_parametrization
    def wrapper(self: Sequence, *args: Any, **kwargs: Any) -> Any:
        storage = self._calls if self._building else self._to_build_calls
        func(self, *args, **kwargs)
        storage.append(_Call(func.__name__, args, kwargs))

    return cast(F, wrapper)


def check_allow_qubit_index(func: F) -> F:
    """Checks if using qubit indices is allowed."""

    @wraps(func)
    def wrapper(self: Sequence, *args: Any, **kwargs: Any) -> Any:
        if not self.is_parametrized() and self.is_register_mappable():
            raise RuntimeError(
                f"Sequence.{func.__name__} cannot be called in"
                " non-parametrized sequences using a mappable register."
            )
        func(self, *args, **kwargs)

    return cast(F, wrapper)


def mark_non_empty(func: F) -> F:
    """Marks the sequence as non-empty."""

    @wraps(func)
    def wrapper(self: Sequence, *args: Any, **kwargs: Any) -> Any:
        func(self, *args, **kwargs)
        self._empty_sequence = False

    return cast(F, wrapper)


def block_if_measured(func: F) -> F:
    """Blocks the call if the sequence has been measured."""

    @wraps(func)
    def wrapper(self: Sequence, *args: Any, **kwargs: Any) -> Any:
        if self.is_measured():
            raise RuntimeError(
                "The sequence has been measured, no further "
                "changes are allowed."
            )
        func(self, *args, **kwargs)

    return cast(F, wrapper)
