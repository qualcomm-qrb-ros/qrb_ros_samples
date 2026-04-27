# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from pathlib import Path
from setuptools import find_packages, setup

package_name = "grasp_perception"
launch_files = [str(p) for p in Path("launch").glob("*.launch.py")]
model_files = [str(p) for p in Path("models").rglob("*") if p.is_file()]
config_files = [str(p) for p in Path("config").glob("*.yaml")]

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", launch_files),
        (f"share/{package_name}/config", config_files),
        *[
            (f"share/{package_name}/{Path(f).parent.as_posix()}", [f])
            for f in model_files
        ],
    ],
    install_requires=["setuptools", "numpy", "opencv-python", "onnxruntime", "pyyaml"],
    zip_safe=True,
    maintainer="Teng Fan",
    maintainer_email="tengf@qti.qualcomm.com",
    description="ROS2 Jazzy DenseFusion RGB-D inference package",
    license="BSD-3-Clause-Clear",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "grasp_perception_node = grasp_perception.grasp_perception_node:main",
            "file_input_publisher = grasp_perception.file_input_publisher:main",
        ],
    },
)
