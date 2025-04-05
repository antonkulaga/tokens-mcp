#!/usr/bin/env python
"""
Setup script for backwards compatibility with pip and tools that don't yet
support pyproject.toml.
"""
from setuptools import setup

# Empty setup script as all configuration is in pyproject.toml
# This ensures the src directory is properly recognized
setup(
    package_dir={"": "src"},
) 