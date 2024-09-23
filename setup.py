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

from setuptools import setup

# Reads the version from the VERSION.txt file
with open("VERSION.txt", "r", encoding="utf-8") as f:
    __version__ = f.read().strip()

if "dev" in __version__:
    raise RuntimeError(
        "The 'pulser' distribution can only be installed or packaged for "
        "stable versions. To install the full development version, run "
        "`make dev-install` instead."
    )

# Pulser packages not pinned to __version__
requirements = [
    "pulser-pasqal",
]
# Adding packages pinned to __version__
with open("packages.txt", "r", encoding="utf-8") as f:
    requirements += [f"{pkg.strip()}=={__version__}" for pkg in f.readlines()]

# Just a meta-package that requires all pulser packages
setup(
    name="pulser",
    version=__version__,
    install_requires=requirements,
    extras_require={"torch": [f"pulser-core[torch] == {__version__}"]},
    description="A pulse-level composer for neutral-atom quantum devices.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Pulser Development Team",
    python_requires=">=3.8",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
    ],
    url="https://github.com/pasqal-io/Pulser",
    zip_safe=False,
)
