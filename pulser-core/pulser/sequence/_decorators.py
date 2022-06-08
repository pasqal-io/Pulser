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

from collections.abc import Callable, Iterable
from functools import wraps
from itertools import chain
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pulser.parametrized import Parametrized
from pulser.sequence._containers import _Call

if TYPE_CHECKING:  # pragma: no cover
    from pulser.sequence.sequence import Sequence


F = TypeVar("F", bound=Callable)


def _screen(func: F) -> F:
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


def _verify_variable(seq: Sequence, x: Any) -> None:
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
    elif isinstance(x, Iterable) and not isinstance(x, str):
        # Recursively look for parametrized objs inside the arguments
        for y in x:
            _verify_variable(seq, y)


def _verify_parametrization(func: F) -> F:
    """Checks and updates the sequence status' consistency with the call.

    - Checks the sequence can still be modified.
    - Checks if all Parametrized inputs stem from declared variables.
    """

    @wraps(func)
    def wrapper(self: Sequence, *args: Any, **kwargs: Any) -> Any:
        if self._is_measured and self.is_parametrized():
            raise RuntimeError(
                "The sequence has been measured, no further "
                "changes are allowed."
            )
        for x in chain(args, kwargs.values()):
            _verify_variable(self, x)
        func(self, *args, **kwargs)

    return cast(F, wrapper)


def _store(func: F) -> F:
    """Checks and stores the call to call it when building the Sequence."""

    @wraps(func)
    @_verify_parametrization
    def wrapper(self: Sequence, *args: Any, **kwargs: Any) -> Any:
        storage = self._calls if self._building else self._to_build_calls
        func(self, *args, **kwargs)
        storage.append(_Call(func.__name__, args, kwargs))

    return cast(F, wrapper)
