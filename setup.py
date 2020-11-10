from setuptools import setup, find_packages

setup(
    name="pulser",
    version="0.0.1",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
    ],
    packages=find_packages(),
    include_package_data=True,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pasqal",
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
