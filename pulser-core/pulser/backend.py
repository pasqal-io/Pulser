"""Base classes for backend execution."""
from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any

from pulser.sequence import Sequence


class CloudConnection:
    pass


class Backend(ABC):
    """The backend abstract base class."""

    @abstractmethod
    def __init__(sequence: Sequence, **kwargs: Any):
        pass

    @abstractmethod
    def run(self, **kwargs) -> Any:
        pass
