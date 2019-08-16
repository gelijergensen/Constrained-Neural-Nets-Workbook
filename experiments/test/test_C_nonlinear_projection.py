import glob
import os

import torch
import torch.nn as nn

from ..C_nonlinear_projection.constraints import (
    helmholtz_equation,
    pythagorean_equation,
)
from ..C_nonlinear_projection.main import run_experiment


def test_proof_of_constraint():

    CHECKPOINT_DIR = ".temp"
    directory = os.path.join(CHECKPOINT_DIR, "test_nonlinear_projection")
    save_file_base = "quick-test"

    # Delete any files that somehow were left over in this directory
    files = glob.glob(f"{directory}/{save_file_base}*.pth")
    for f in files:
        os.remove(f)

    all_files = list()
    failure = None
    for method in ["unconstrained", "soft-constrained"]:

        for constraint, constraint_name in zip(
            [helmholtz_equation, pythagorean_equation],
            ["helmholtz", "pythagorean"],
        ):

            save_file = f"{save_file_base}_{method}_{constraint_name}"

            num_epochs = 1
            final_result = run_experiment(
                num_epochs,
                save_directory=directory,
                save_file=save_file,
                method=method,
                constraint=constraint,
                max_iterations=10,
                tolerance=0,  # TODO relax this once we fix the bug
            )

            # Try to load in the model again
            files = glob.glob(f"{directory}/{save_file}*.pth")
            all_files.extend(files)

            try:
                assert len(files) == num_epochs

                loaded_result = torch.load(files[-1])

                loaded_config = loaded_result["configuration"]
                final_config = final_result[0]

                # Can't compare the functions directly
                assert type(loaded_config.pop("model_act")) == type(
                    final_config.pop("model_act")
                )
                assert (
                    loaded_config == final_config
                )  # Remainder compared directly

            except AssertionError as assertFailed:
                failure = assertFailed
            else:
                failure is None

    # cleanup
    for f in all_files:
        os.remove(f)

    if failure is not None:
        raise failure
