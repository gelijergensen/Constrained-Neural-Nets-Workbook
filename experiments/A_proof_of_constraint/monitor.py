"""Things to monitor during the training and evaluation phases"""

import numpy as np
import torch

from pyinsulate.handlers import Monitor


class ProofOfConstraintMonitor(Monitor):
    def __init__(self, is_evaluation=False):
        super().__init__()
        self.is_evaluation = is_evaluation

        # epoch and iteration are recorded automatically
        self.add_key("batch_size")
        self.add_key("loss_percentiles")
        self.add_key("loss_abs_percentiles")
        self.add_key("mean_loss")
        self.add_key("constrained_loss")
        self.add_key("constraints_percentiles")
        self.add_key("constraints_abs_percentiles")
        self.add_key("reduced_constraints")
        # self.add_key("constraints_diagnostics")
        self.add_key("model_parameters")
        self.add_key("model_parameters_grad")
        self.add_key("timing")
        self.timing_keys = None

    def summarize(self):
        mean_loss = np.mean(self.mean_loss[-1])
        mean_constrained_loss = np.mean(self.constrained_loss[-1])
        summary = f"Mean constrained loss: {mean_constrained_loss:0.5f}, Mean loss: {mean_loss:0.5f}"
        return summary

    @staticmethod
    def get_tensor(tensor):
        return tensor.cpu().detach()

    @staticmethod
    def get_tensor_item(tensor):
        return tensor.cpu().item()

    def finalize(self, engine):
        self.add_value("batch_size", np.array(self.ctx["batch_size"]))
        self.add_value(
            "loss_percentiles",
            np.percentile(
                torch.cat(self.ctx["loss"], dim=0).numpy(),
                np.linspace(0, 100, num=101),
            ),
        )
        self.add_value(
            "loss_abs_percentiles",
            np.percentile(
                torch.abs(torch.cat(self.ctx["loss"], dim=0)).numpy(),
                np.linspace(0, 100, num=101),
            ),
        )
        self.add_value(
            "mean_loss",
            np.average(
                np.array(self.ctx["mean_loss"]),
                weights=np.array(self.ctx["batch_size"]),
            ),
        )
        self.add_value(
            "constrained_loss",
            np.average(
                np.array(self.ctx["constrained_loss"]),
                weights=np.array(self.ctx["batch_size"]),
            ),
        )
        self.add_value(
            "constraints_percentiles",
            np.percentile(
                torch.cat(self.ctx["constraints"], dim=0).numpy(),
                np.linspace(0, 100, num=101),
            ),
        )
        self.add_value(
            "constraints_abs_percentiles",
            np.percentile(
                torch.abs(torch.cat(self.ctx["constraints"], dim=0)).numpy(),
                np.linspace(0, 100, num=101),
            ),
        )
        self.add_value(
            "reduced_constraints",
            np.average(
                np.array(self.ctx["reduced_constraints"]),
                weights=np.array(self.ctx["batch_size"]),
            ),
        )
        # I don't know whether I want to keep this, really
        # self.add_value("constraints_diagnostics", ...)
        if self.is_evaluation:
            self.add_value(
                "model_parameters",
                np.percentile(
                    self.ctx["model_parameters"][0].numpy(),
                    np.linspace(0, 100, num=101),
                ),
            )
        else:
            # Percentile over the entire epoch and all parameters
            self.add_value(
                "model_parameters",
                np.percentile(
                    torch.stack(self.ctx["model_parameters"], dim=0).numpy(),
                    np.linspace(0, 100, num=101),
                ),
            )
            self.add_value(
                "model_parameters_grad",
                np.percentile(
                    torch.stack(
                        self.ctx["model_parameters_grad"], dim=0
                    ).numpy(),
                    np.linspace(0, 100, num=101),
                ),
            )

        # only the items with full batches
        timing_mask = (
            np.array(self.ctx["batch_size"])
            == np.array(self.ctx["batch_size"])[0]
        )
        timing_dict = dict()
        for key in self.timing_keys:
            times = np.array(
                [
                    self.ctx["timing"][i][key]
                    for i in range(len(self.ctx["timing"]))
                ]
            )
            timing_dict[key] = np.average(
                times[np.logical_and(timing_mask, times > -998)]
            )
        self.add_value("timing", timing_dict)

    def __call__(self, engine):
        self.ctx["batch_size"].append(len(self.get_tensor(engine.state.xb[0])))
        self.ctx["loss"].append(self.get_tensor(engine.state.loss))
        self.ctx["mean_loss"].append(
            self.get_tensor_item(engine.state.mean_loss)
        )
        self.ctx["constrained_loss"].append(
            self.get_tensor_item(engine.state.constrained_loss)
        )
        self.ctx["constraints"].append(
            self.get_tensor(engine.state.constraints)
        )
        # Only true because single constraint!
        self.ctx["reduced_constraints"].append(
            self.get_tensor_item(engine.state.reduced_constraints)
        )
        self.ctx["constraints_diagnostics"].append(
            map(self.get_tensor, engine.state.constraints_diagnostics)
        )
        if self.is_evaluation:
            if len(self.ctx["model_parameters"]) == 0:
                self.ctx["model_parameters"].append(
                    self.get_tensor(engine.state.model_parameters)
                )
        else:
            self.ctx["model_parameters"].append(
                self.get_tensor(engine.state.model_parameters)
            )
            self.ctx["model_parameters_grad"].append(
                self.get_tensor(engine.state.model_parameters_grad)
            )
        self.ctx["timing"].append(engine.state.times.copy())
        if self.timing_keys is None:
            self.timing_keys = list(engine.state.times.keys())

