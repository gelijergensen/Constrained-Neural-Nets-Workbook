"""Runs a single copy of the experiment for a particular configuration"""

from datetime import datetime
import sys
import torch

from experiments.A_proof_of_constraint.main import run_experiment
from experiments.A_proof_of_constraint.experiment_definition import get_configuration

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify a configuration!")
        sys.exit()
    idx = int(sys.argv[1])

    configuration = get_configuration(idx)

    directory = "/global/u1/g/gelijerg/Projects/pyinsulate/results"
    base_name = f"proof-of-constraint_{idx:03d}"
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    savefile = f"{base_name}_{time_string}.pth"

    # Default to 1 epoch, if not specified
    num_epochs = configuration.pop('num_epochs', 1)

    final_result = run_experiment(num_epochs, log=print, **configuration)
    configuration, engines, monitors = final_result
    trainer, train_evaluator, test_evaluator = engines
    training_monitor, evaluation_train_monitor, evaluation_test_monitor = monitors

    summary = {
        'configuration': configuration,
        'training_monitor': training_monitor,
        'evaluation_train_monitor': evaluation_train_monitor,
        'evaluation_test_monitor': evaluation_test_monitor,
    }
    full_file = f"{directory}/{savefile}"
    print(f"Saving to file {full_file}")
    torch.save(summary, full_file)

    print('done!')
