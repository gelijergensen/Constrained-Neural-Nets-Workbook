import numpy as np
import torch

if __name__ == "__main__":

    loadfile = "hyper-poc_2019-07-02-11-45-47.pth"
    # loadfile = "hyper-poc_2019-07-02-11-47-48.pth"

    with open(loadfile, "rb") as f:
        results = torch.load(f)

    # First, print the best by sum of losses
    def sum_of_losses(result):
        return sum(result[1])

    idxs = np.argsort([sum_of_losses(result) for result in results])

    print(idxs)

    # Print best 10
    for i in idxs[:10]:
        print(results[i])

    print(len(results))
    print(len(results) / len(np.unique([result[1][0] for result in results])))
