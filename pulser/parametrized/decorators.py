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
from functools import wraps
from itertools import chain

from pulser.parametrized import Parametrized, ParamObj


def parametrize(func):
    """Makes a function support parametrized arguments.

    Note:
        Designed for use in class methods. Usage in instance or static methods
        is not supported, and in regular functions is not tested.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        for x in chain(args, kwargs.values()):
            if isinstance(x, Parametrized):
                return ParamObj(func, *args, **kwargs)
        return func(*args, **kwargs)

    return wrapper
