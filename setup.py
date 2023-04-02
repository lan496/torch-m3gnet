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
    "torch",
    # Trouble in enabling grad in evaluation at pytorch-lightning==1.7.x
    "pytorch-lightning==1.8.6",
    "torch-scatter",
    "torch-sparse",
    "torch-geometric",
    "torchmetrics",
    "pymatgen>=2022.7.25",
    "ruamel.yaml",
    "joblib",
    "scikit-learn",
]

EXTRAS = {
    "dev": [
        "pytest",
        "pytest-cov==4.0.0",
        "typeguard==2.13.3",
        "pre-commit",
        "black",
        "mypy",
        "flake8",
        "pyupgrade",
        "pydocstyle",
        "nbqa",
        "torchinfo",
        "notebook",
        "ipython",
        "ipykernel",
        "seaborn",
    ],
    "docs": [
        # We cannot update sphinx to >6 until docutils==0.20 is released: https://github.com/mcmtroffaes/sphinxcontrib-bibtex/issues/322
        "docutils==0.17",
        "sphinx<6",
        "sphinx-autobuild",
        # "nbsphinx",
        "sphinxcontrib-bibtex",
        "myst-parser",
        "sphinx-book-theme",
        "linkify-it-py",
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
