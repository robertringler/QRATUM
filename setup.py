"""QuASIM setup script."""

from setuptools import find_packages, setup

setup(
    name="quasim",
    version="0.1.0",
    description="Quantum-Accelerated Simulation Runtime",
    author="Sybernix Team",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "quasim-hcal=quasim.cli.main:main",
        ],
    },
    python_requires=">=3.8",
)
