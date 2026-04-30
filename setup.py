from setuptools import find_packages, setup

setup(
    name="py-risk-simulation",
    version="0.1.0",
    description="Simplified Murex-style risk pipeline simulation in Python",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "matplotlib>=3.8",
        "yfinance>=0.2",
        "scipy>=1.11",
    ],
    extras_require={"dev": ["pytest>=8.0", "jupyter>=1.0"]},
    python_requires=">=3.10",
)
