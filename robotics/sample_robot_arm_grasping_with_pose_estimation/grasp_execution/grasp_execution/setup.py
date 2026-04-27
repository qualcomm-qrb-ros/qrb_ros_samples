# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Setuptools configuration for the grasp_execution application layer."""

from setuptools import find_namespace_packages, setup


setup(
    name="grasp_execution",
    version="1.0.0",
    description=(
        "Vision-guided robot arm grasping: RM API control, ROS2 pose subscription, "
        "and grasp state machine."
    ),
    author="Teng Fan",
    author_email="tengf@qti.qualcomm.com",
    license="BSD-3-Clause-Clear",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.9",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "grasp-execution=core.grasp_execution:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="robotics, grasping, pose estimation, RM arm",
)
