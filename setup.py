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

setup(
    name="pulser",
    version="0.0.1a1",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
    ],
    packages=find_packages(),
    include_package_data=True,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pulser Development Team",
    license="Apache 2.0",
    classifiers=[
         "Development Status :: 2 - Pre-Alpha",
         "Programming Language :: Python :: 3",
         "Operating System :: MacOS",
         "Operating System :: Unix",
         "Operating System :: Microsoft :: Windows",
         "Topic :: Scientific/Engineering",
         ],
    url="https://pasqal.io/",
)
