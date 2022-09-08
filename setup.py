import setuptools
from pathlib import Path

with open(Path("./README.md").resolve(), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "lymph", # Replace with your own username
    version = "0.1.1",
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