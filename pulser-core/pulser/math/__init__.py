# Copyright 2024 Pulser Development Team
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

"""Custom implementation of math and array functions."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pulser.math.abstract_array import (
    AbstractArray as AbstractArray,
    AbstractArrayLike,
)

if TYPE_CHECKING:
    import torch

# Custom function definitions


def sin(a: AbstractArrayLike, /) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.sin(a.as_tensor()))
    return AbstractArray(np.sin(a.as_array()))
