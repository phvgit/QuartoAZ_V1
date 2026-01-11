#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for compiling the C++ MCTS module.

Usage:
    cd alphaquarto/ai/mcts_cpp
    pip install pybind11
    python setup.py build_ext --inplace

Or from project root:
    pip install -e .
"""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class get_pybind_include:
    """Helper class to determine the pybind11 include path."""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'mcts_cpp',
        sources=['bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
        ],
        language='c++',
        extra_compile_args=['-O3', '-std=c++17', '-fPIC'],
    ),
]

class BuildExt(build_ext):
    """Custom build extension to handle C++17."""
    def build_extensions(self):
        # Compiler-specific flags
        if self.compiler.compiler_type == 'unix':
            for ext in self.extensions:
                ext.extra_compile_args = ['-O3', '-std=c++17', '-fPIC', '-Wall']
        elif self.compiler.compiler_type == 'msvc':
            for ext in self.extensions:
                ext.extra_compile_args = ['/O2', '/std:c++17']

        build_ext.build_extensions(self)

setup(
    name='mcts_cpp',
    version='1.0.0',
    author='AlphaQuarto',
    description='Fast MCTS implementation in C++ for AlphaZero Quarto',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.8',
)
