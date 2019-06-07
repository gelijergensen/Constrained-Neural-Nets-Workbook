import torch


def get_devices(log=print):
    """Retrieves a pair of (cpu, gpu). If there is no gpu, then gpu = cpu"""
    if torch.cuda.is_available():
        # Note, since this might not happen, there's a chance the device called
        # "gpu" is actually the cpu
        gpu = torch.device("cuda")
        log("There is a gpu available:", gpu)
    else:
        gpu = torch.device("cpu")
        log("No gpu available:", gpu)
    cpu = torch.device("cpu")
    return cpu, gpu
