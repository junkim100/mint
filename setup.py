"""Setup script for MINT."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="mint",
    version="0.1.0",
    description="Risk-calibrated tool selection for LLMs using mechanistic interpretability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jun Kim",
    author_email="junkim100@users.noreply.github.com",
    url="https://github.com/junkim100/mint",
    packages=find_packages(exclude=["tests", "scripts", "configs"]),
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm tool-use mechanistic-interpretability conformal-prediction",
)

