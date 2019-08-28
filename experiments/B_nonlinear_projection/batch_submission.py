"""Runs a single copy of the experiment for a particular configuration"""

from datetime import datetime
import os
import sys
import torch

from experiments.B_nonlinear_projection.main import run_experiment
from experiments.B_nonlinear_projection.experiment_definition import (
    get_configuration,
)


def get_savefile():
    base_name = "nonlinear-projection"
    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    savefile = f"{base_name}_{time_string}.pth"
    return savefile


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify a configuration!")
        sys.exit()
    idx = int(sys.argv[1])

    configuration = get_configuration(idx)
    num_epochs = configuration.pop("num_epochs")

    savefile = get_savefile()
    save_directory = os.path.expandvars(
        "$SCRATCH/results/checkpoints/B_nonlinear_projection"
    )
    save_interval = configuration.get("save_interval", None)
    print(f"Running proof of constraint with savefile {savefile}")
    checkpoint_save_file_base = os.path.splitext(savefile)[0]
    final_checkpoint = f"{checkpoint_save_file_base}_{num_epochs:05d}.pth"

    final_result = run_experiment(
        num_epochs,
        log=print,
        save_directory=save_directory,
        save_file=checkpoint_save_file_base,
        save_interval=save_interval,
        evaluate=True,
        projection=True,
        **configuration,
    )
    print(f"Completed run with savefile {savefile}")
    print(f"Checkpoint was saved to {save_directory}//{final_checkpoint}")

    print("done!")
