# Caution Regarding the Spectral-Model Branch

This version of the code was an initial foray into using Spectral Methods to represent the functions. However, this was not fully implemented, as it was deemed not promising enough to continue. I have included the code here in the off chance that I or someone else wishes to use it in future work.

# Constrained Neural Nets Workbook [![Build Status](https://travis-ci.com/gelijergensen/Constrained-Neural-Nets-Workbook.svg?token=JdexHmEcyj7BDKQEoi8S&branch=master)](https://travis-ci.com/gelijergensen/Constrained-Neural-Nets-Workbook) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
A workbook for examining some ideas for constraining neural networks in PyTorch.

# [Github Page](https://gelijergensen.github.io/Constrained-Neural-Nets-Workbook/)

This workbook has a [Github Page](https://gelijergensen.github.io/Constrained-Neural-Nets-Workbook/) where I describe the course of the experiments and show some resultant figures.

# Project Structure

The project is divided into 3 main directories: `experiments/`, `src/`, and `test/`. These are described briefly below, but you can also examine the `__init__.py` files in the root of each directory for a more detailed description of the directory structure. Additionally, there are a couple other directories which are simply for support. These are also described below.

## `experiments/`

This directory contains the code specific to running particular experiments. Each experiment has its own subdirectory which contains all the code specific to that experiment. Additionally, in the top of this directory, there are a number of Jupyter Notebooks for running and visualizing the experiments.

## `src/`

This directory contains the code which is common to multiple or all experiments. For the most part, this is base classes which are extended for use in a particular experiment.

## `test/`

This directory follows a nearly identical structure to `src/`. All files which are prepended with "test_" are test files corresponding to a particular source file (located in the same spot in the directory structure). All tests in this directory are run by the `pre-push` hook (see [Best Practices](#best-practices) below).

## `slurm/`

This directory houses some shell scripts necessary for submitting slurm jobs which will run the experiments on a batch system which uses slurm. These have only been tested on a single slurm system, so they may not work in general.

## `docs/`

Here I have placed the markdown and supporting files which are hosted on the [Github Page](https://gelijergensen.github.io/Constrained-Neural-Nets-Workbook/).

# Developing

You may feel free to use this code in any way you like. Below, we detail the setup process for using this repository directly and also describe some of the best practices used in developing this code.

## Setup
It is recommended that you develop on a conda virtual environment, even though we do actually run our tests in a minimal `pip` environment. The easiest way to set this up is as follows:

```bash
conda create -n yourenvname python=3.6 anaconda
source activate yourenvname # activate env. Deactivate can be done with `conda deactivate`
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install pytorch-ignite # Pip package necessary for running the code
```

This can take around 20 minutes. Further, you need to ensure that the environment variable "$SCRATCH" is set. If it is not already set in the system, you can add it to the definition of the conda environment:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export SCRATCH=<desired scratch location>" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo "unset SCRATCH" > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

Remember to re-activate the virtual environment for the changes to take place. From here, it is helpful to install a couple _optional_ packages:


```bash
pip install pytest # For running tests (see Best Practices below)
pip install clean_ipynb # For cleaning up Jupyter notebooks (see Best Practices below)
pip install black # For formatting (see Best Practices below)
```

Once you have everything installed, you can install the remaining requirements with
```bash
pip install -e .
```
called from the root directory of this project.

Lastly, you will want to load the data for running unit tests. This can be done using the script provided in the root of this project:

```bash
source loadTestData.sh
```

At this point, you should test that you have everything successfully installed with a quick

```bash
pytest
```

This will run the test suite. Primarily, you want to make sure that the file `test/test_imports.py` passes.

## Best Practices
In order to produce the smallest possible diffs and to speed up development, we suggest a few best practices:

### Use the pre-push githook to run tests locally

We already have provided the pre-push githook which runs the test suite before allowing a `git push` command. You can enable this hook by adding the hooks path to your local git configuration:
```bash
git config --local core.hookspath .githooks
```
If, for any reason, you need to perform a push without running the test suite, you can always use `git push --no-verify`

### Work on feature branches

We generally follow the [Git Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow). The general idea here is to create a local branch for each individual feature (`git checkout -b <branchname>`), work on that feature on your local branch and then push to the remote with `git push origin <branchname>`. From here, you can make a pull request and we can run tests on the new branch before squashing the commits into a single commit and merging them into the master branch. In this way, the master branch should always contain a working version of the code which passes the tests.

### Clean and format code

Jupyter notebooks can easily become very large when you keep the saved variables in them. Fortunately, there is a simple `pip` package which allows for the cleaning of notebooks ([clean_ipynb](https://github.com/KwatME/clean_ipynb)). Assuming you have this installed, you can simply run the clean script in the base of the repository (`source clean`), which will clean up all of the notebooks in the experiment directory. We recommend that you do this _before_ you perform a commit locally.

We also use the [black autoformatter](https://pypi.org/project/black/) to ensure that the style of the code is consistent and to minimize the size of the diff files. We especially recommend here that you configure your IDE/code editor to automatically format on save. If you are using [VS Code](https://code.visualstudio.com/) and already have black installed (`pip install black`), you will want to add the following to your configuration:

```json
"editor.formatOnSave": true,
"python.formatting.provider": "black",
"python.formatting.blackArgs": [
    "--line-length",
    "80"
  ],
```