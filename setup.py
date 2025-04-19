#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="strata",
    version="0.1.0",
    author="Andrew Vasilyev",
    author_email="",
    description="PBR Texture Exporter for PSD files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/retran/strata",
    packages=["strata"],
    package_dir={"strata": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "strata=strata.strata:main",
        ],
    },
)