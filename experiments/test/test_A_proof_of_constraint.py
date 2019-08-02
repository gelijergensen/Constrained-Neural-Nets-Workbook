import glob
import os

import torch
import torch.nn as nn

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
    for method in [
        "constrained",
        "batchwise",
        "reduction",
        "unconstrained",
        "no-loss",
        "non-projecting",
    ]:
        reduction = None if method != "reduction" else Lp_Reduction(1)
        save_file = f"{save_file_base}_{method}"

        print(reduction)

        num_epochs = 1
        final_result = run_experiment(
            num_epochs,
            save_directory=directory,
            save_file=save_file,
            num_points=20,
            num_training=10,
            batch_size=10,
            method=method,
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
            assert loaded_config == final_config  # Remainder compared directly

        except AssertionError as assertFailed:
            failure = assertFailed
        else:
            failure = None

    # cleanup
    for f in all_files:
        os.remove(f)

    if failure is not None:
        raise failure
