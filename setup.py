import io
import os

from setuptools import find_packages, setup

# Package meta-data.
REQUIRED = [
    "setuptools",
    "setuptools_scm",
    "wheel",
    "typing_extensions",
    "torchtyping==0.1.4",
    "torch>=1.12.0",
    "torch-scatter",
    "torch-sparse",
    "torch-spline-conv",
    "torch-geometric",
    "pymatgen>=2022.7.25",
]

EXTRAS = {
    "dev": [
        "pytest==7.1.3",
        # "pytest-cov==3.0.0",
        "typeguard==2.13.3",
        "pre-commit",
        "black",
        "mypy",
        "flake8",
        "pyupgrade",
        "pydocstyle",
        "nbqa",
    ],
    "docs": [
        "sphinx",
        "sphinx-autobuild",
        # "nbsphinx",
        "sphinxcontrib-bibtex",
        "myst-parser",
        "sphinx-book-theme",
    ],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = ""


# Where the magic happens:
setup(
    name="torch_m3gnet",
    description="PyTorch implementation of M3GNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kohei Shinohara",
    author_email="kshinohara0508@gmail.com",
    python_requires=">=3.8.0",
    url="https://github.com/lan496/torch-m3gnet",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["torch_m3gnet"]),
    package_data={},
    # numpy: https://github.com/numpy/numpy/issues/2434
    setup_requires=["setuptools_scm", "numpy"],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="BSD",
    test_suite="tests",
    zip_safe=False,
    use_scm_version=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
