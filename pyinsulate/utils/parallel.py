"""Tools for running models in PyTorch in parallel"""


def send_all_models_to_gpus(*models):
    if torch.cuda.is_available():
        print("Ooh, a gpu!")
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print("Many gpus!")
            device_list = list(range(num_gpus))
            out_models = [nn.DataParallel(model.to(device), device_list)
                          for model in models]
        else:
            out_models = [model.to(device) for model in models]
    else:
        out_models = models
    return out_models
