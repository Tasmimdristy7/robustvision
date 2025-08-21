"""
Setup script for RobustVision
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="robustvision",
    version="0.1.0",
    author="RobustVision Team",
    author_email="team@robustvision.org",
    description="Adversarial & Reliability Testbench for Vision Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robustvision/robustvision",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "robustvision=robustvision.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "robustvision": ["configs/*.yaml"],
    },
    keywords="machine-learning computer-vision adversarial-robustness security testing",
    project_urls={
        "Bug Reports": "https://github.com/robustvision/robustvision/issues",
        "Source": "https://github.com/robustvision/robustvision",
        "Documentation": "https://robustvision.readthedocs.io/",
    },
) 