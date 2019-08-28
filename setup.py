"""Setup file for Constrained-Neural-Nets-Workbook"""

from setuptools import setup

VERSION = "0.0.0.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

PACKAGE_NAMES = ["src"]
KEYWORDS = []
SHORT_DESCRIPTION = (
    "Tools for the training of networks with hard constraints in PyTorch"
)

CLASSIFIERS = [
    "Development Status :: 2 - PreAlpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
]

PACKAGE_REQUIREMENTS = ["numpy", "torch", "torchvision", "pytorch-ignite"]

TEST_REQUIREMENTS = ["pytest", "clean_ipynb"]

if __name__ == "__main__":
    setup(
        name="Constrained-Neural-Nets-Workbook",
        version=VERSION,
        python_requires=">=3",
        description=SHORT_DESCRIPTION,
        author="G. Eli Jergensen",
        author_email="gejergensen@lbl.gov",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="MIT",
        url="https://github.com/gelijergensen/Constrained-Neural-Nets-Workbook",
        packages=PACKAGE_NAMES,
        scripts=[],
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        zip_safe=False,
        install_requires=PACKAGE_REQUIREMENTS,
        tests_require=TEST_REQUIREMENTS,
    )
