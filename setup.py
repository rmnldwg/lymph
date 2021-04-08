import setuptools
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "lymph", # Replace with your own username
    version = "1.0.3",
    author = "Roman Ludwig",
    author_email = "roman.ludwig@usz.ch",
    description = "Package for statistical modelling of lymphatic metastatic spread.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/rmnldwg/lymph",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',
)