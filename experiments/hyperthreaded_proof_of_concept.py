from datetime import datetime
from multiprocessing import cpu_count, current_process
import numpy as np
import os
from proof_of_concept import run_analysis, load_data, create_dataloaders
from utilities.hyperthreading import perform_multiprocessed
import torch
from torch import nn


def initialize(args):

    print(f"Initializing in {current_process()}")

    global train_set, test_set, batch_size, train_dl, test_dl, activations, act_keys

    train_dl, test_dl = create_dataloaders(
        train_set.clone(), test_set.clone(), batch_size
    )

    activations = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'leakyrelu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
        'selu': nn.SELU(),
    }
    act_keys = list(activations.keys())
    act_keys.sort()

    print(f"Initialized {current_process()}")


def multiprocessed_analysis(args):
    # Args is just the iteration number, which we don't really care about

    global train_dl, test_dl, activations, act_keys  # initialized in main

    # Pick some random shape and activation function
    length = np.random.randint(3, 15)
    shape = np.random.randint(1, 25, size=length) * 10
    act_idx = np.random.randint(len(act_keys))
    act = activations[act_keys[act_idx]]

    losses = run_analysis(train_dl, test_dl, shape, act)

    return ((shape, act), losses)
    # return args


def finalize(results):
    global save_file  # created by the main call

    with open(savefile, "wb") as f:
        torch.save(results, savefile)
    return results


if __name__ == "__main__":

    base_name = "hyper-poc"
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    savefile = f"{base_name}_{time_string}.pth"

    print(f"Running hyperthreaded POC with savefile {savefile}")

    # 4 runs
    nproc = 2
    num_jobs = nproc * 4

    print(f'num_jobs: {num_jobs}')

    args = ((i,) for i in range(num_jobs))

    path = os.path.expandvars('$SCRATCH/data/divfree-test/raw_0100.npy')
    batch_size = 128
    train_set, test_set = load_data(
        path, 32, batch_size=batch_size, just_datasets=True)

    results = perform_multiprocessed(
        initialize, (None,), multiprocessed_analysis, args, finalize, nproc=nproc
    )

    print(results)
    print(f"Saved to file {savefile}")
    print('done!')
