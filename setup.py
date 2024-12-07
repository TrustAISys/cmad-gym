#!/usr/bin/env python

import os

from setuptools import find_packages, setup

# Prepare long description using existing docs
long_description = ""
this_dir = os.path.abspath(os.path.dirname(__file__))
doc_files = ["README.md"]
for doc in doc_files:
    with open(os.path.join(this_dir, doc), "r") as f:
        long_description = "\n".join([long_description, f.read()])

setup(
    name="cmad-gym",
    version="0.1.0",
    description="a High-level Customizable Multi-agent Gym for Dependable Autonomous Driving",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TrustAISys/cmad-gym",
    author="Morphlng",
    author_email="morphlng@proton.me",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "GPUtil",
        "Shapely>=1.7.1",
        "carla",
        "ephem",
        "gym<=0.22.0",
        "networkx>=2.2",
        "numpy<=1.23.4",
        "opencv-python",
        "orjson",
        "psutil",
        "py-trees~=0.8.3",
        "pygame<=2.1.0",
        "redis",
        "simple-watchdog-timer",
        "six",
        "tabulate",
        "xmlschema>=1.0.18",
    ],
    extras_require={
        "test": ["tox", "pytest", "pytest-xdist"],
        "ray": [
            "ray==1.8.0",
            "ray[tune]==1.8.0",
            "ray[rllib]==1.8.0",
            "protobuf<3.21.0",
        ],
    },
    keywords="multi-agent learning environments OpenAI Gym CARLA",
    project_urls={
        "Source": "https://github.com/TrustAISys/cmad-gym",
        "Report bug": "https://github.com/TrustAISys/cmad-gym/issues",
        "Author website": "https://github.com/TrustAISys",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
