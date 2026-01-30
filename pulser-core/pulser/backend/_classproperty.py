"""Defines a custom descriptor for a class property."""

from __future__ import annotations

from typing import Any, Type, TypeVar

T = TypeVar("T")


class classproperty:
    """A custom descriptor to make a class property."""

    def __init__(self, fget: Any):  # noqa: D107
        self.fget = fget

    def __get__(self, instance: T, owner: Type[T]) -> Any:
        return self.fget(owner)
