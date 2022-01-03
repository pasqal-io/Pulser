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

from setuptools import setup, find_packages

__version__ = ""
exec(open("pulser/_version.py").read())

setup(
    name="pulser",
    version=__version__,
    install_requires=[
        "matplotlib",
        "numpy>=1.20, <1.22",
        "scipy",
        "qutip",
    ],
    extras_require={
        ":python_version == '3.7'": [
            "backports.cached-property",
            "typing-extensions",
        ],
    },
    packages=find_packages(),
    include_package_data=True,
    description="A pulse-level composer for neutral-atom quantum devices.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pulser Development Team",
    python_requires=">=3.7.0",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
    ],
    url="https://github.com/pasqal-io/Pulser",
)
