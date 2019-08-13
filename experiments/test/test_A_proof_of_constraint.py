import glob
import os

import torch
import torch.nn as nn

from ..A_proof_of_constraint.constraints import (
    helmholtz_equation,
    pythagorean_equation,
)
from ..A_proof_of_constraint.main import run_experiment
from ..A_proof_of_constraint.reductions import Lp_Reduction


def test_proof_of_constraint():

    CHECKPOINT_DIR = ".temp"
    directory = os.path.join(CHECKPOINT_DIR, "test_proof_of_constraint")
    save_file_base = "quick-test"

    # Delete any files that somehow were left over in this directory
    files = glob.glob(f"{directory}/{save_file_base}*.pth")
    for f in files:
        os.remove(f)

    all_files = list()
    failure = None
    for method in [
        "constrained",
        "batchwise",
        "reduction",
        "unconstrained",
        "soft-constrained",
        "no-loss",
        "non-projecting",
    ]:
        reduction = Lp_Reduction(1) if method in ["reduction"] else None

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
                reduction=reduction,
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
                # Can't compare the reductions directly
                assert type(loaded_config.pop("reduction")) == type(
                    final_config.pop("reduction")
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
