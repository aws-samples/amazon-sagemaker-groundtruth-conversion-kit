# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Open source library to convert SageMaker Ground Truth manifests into standard formats."""
import os
from setuptools import setup
from setuptools import find_packages
from pathlib import Path


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_version():
    return read("VERSION").strip()


# Specific use case dependencies
extras = {"test": (["flake8", "pytest", "black", "flaky"],)}


setup(
    name="gt_conversion",
    version=read_version(),
    description=__doc__,
    packages=find_packages(),
    long_description=read("README.md"),
    author="Amazon Web Services",
    url="https://github.com/aws/",  # TODO: Fix this when known
    license="Apache License 2.0",
    keywords="Sagemaker",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=Path("requirements.txt").read_text().splitlines(),
    extras_require=extras,
)
