from datetime import datetime
import torch

from A_proof_of_constraint.main import run_experiment


def get_savefile():
    base_name = "proof-of-constraint"
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    savefile = f"{base_name}_{time_string}.pth"
    return savefile


def save_out(summary, savefile, directory="/global/u1/g/gelijerg/Projects/pyinsulate/results"):
    full_file = f"{directory}/{savefile}"
    print(f"Saving to file {full_file}")
    torch.save(summary, full_file)


# TODO this should be converted to a Jupyter Notebook probably

if __name__ == "__main__":

    savefile = get_savefile()
    print(f"Running proof of constraint with savefile {savefile}")

    final_result = run_experiment(
        1, log=print,
        **{
            'training_sampling': "uniform",
            'num_points': 100,
            'num_training': 10,
            'batch_size': 10,
            'model_size': [20, 20, 20],
            'learning_rate': 1e-2,
        })

    configuration, (training_monitor, evaluation_train_monitor,
                    evaluation_test_monitor) = final_result

    save_out({
        'configuration': configuration,
        'training_monitor': training_monitor,
        'evaluation_train_monitor': evaluation_train_monitor,
        'evaluation_test_monitor': evaluation_test_monitor,
    }, savefile=savefile)
    print('Done!')
