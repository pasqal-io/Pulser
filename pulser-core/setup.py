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
import os
from pathlib import Path

from setuptools import find_packages, setup

distribution_name = "pulser-core"  # The name on PyPI
package_name = "pulser"  # The main module name
current_directory = Path(__file__).parent

# Reads the version from the VERSION.txt file
with open(current_directory.parent / "VERSION.txt", "r") as f:
    __version__ = f.read().strip()

# Changes to the directory where setup.py is
os.chdir(current_directory)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Stashes the source code for the local version file
local_version_fpath = Path(package_name) / "_version.py"
with open(local_version_fpath, "r") as f:
    stashed_version_source = f.read()

# Overwrites the _version.py for the source distribution (reverted at the end)
with open(local_version_fpath, "w") as f:
    f.write(f"__version__ = '{__version__}'\n")

setup(
    name=distribution_name,
    version=__version__,
    install_requires=requirements,
    packages=find_packages(),
    package_data={package_name: ["py.typed"]},
    include_package_data=True,
    description="A pulse-level composer for neutral-atom quantum devices.",
    long_description=open("README.md").read(),
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

# Restores the original source code of _version.py
with open(local_version_fpath, "w") as f:
    f.write(stashed_version_source)
