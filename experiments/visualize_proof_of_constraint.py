import os
import sys
import torch


from A_proof_of_constraint.visualize import plot_loss, plot_constraints, plot_time


def load(directory, loadfile):
    full_file = f"{directory}/{loadfile}"
    print(f"loading from {full_file}")
    return torch.load(full_file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify a script!")
        sys.exit()
    path = sys.argv[1]
    if len(sys.argv) > 2:
        directory = sys.argv[2]
    else:
        directory = "/global/u1/g/gelijerg/Projects/pyinsulate/results"

    basename = os.path.splitext(os.path.basename(path))[0]

    try:
        summary = load(directory, path)
    except IOError as e:
        print(f"Unable to locate file. Please make sure it is in {directory}")
        raise e

    training_monitor = summary['training_monitor']
    evaluation_train_monitor = summary['evaluation_train_monitor']
    evaluation_test_monitor = summary['evaluation_test_monitor']

    print(training_monitor)
    plot_loss([training_monitor, evaluation_train_monitor, evaluation_test_monitor], [
              "Training", "Evaluation on Training", "Evaluation on Testing"], f"training-loss_{basename}")

    plot_constraints([training_monitor, evaluation_train_monitor, evaluation_test_monitor], [
        "Training", "Evaluation on Training", "Evaluation on Testing"], f"training-constraint_{basename}")

    plot_time(training_monitor, f"compute-time_{basename}")

    print('Done!')
