"""Things to monitor during the training and evaluation phases"""

import numpy as np
import torch

from pyinsulate.handlers import Monitor


class ProofOfConstraintMonitor(Monitor):
    def __init__(self, is_evaluation=False):
        super().__init__(is_evaluation=is_evaluation)

        # epoch and iteration are recorded automatically
        self.add_key("batch_size")
        self.add_key("loss")
        self.add_key("mean_loss")
        self.add_key("constrained_loss")
        self.add_key("constraints")
        self.add_key("reduced_constraints")
        self.add_key("constraints_diagnostics")
        self.add_key("model_parameters")
        self.add_key("model_parameters_grad")
        self.add_key("timing")

    def summarize(self):
        mean_loss = np.mean(self.get("mean_loss", -1))
        mean_constrained_loss = np.mean(self.get("constrained_loss", -1))
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
            "loss",
            np.percentile(
                torch.cat(self.ctx["loss"], dim=0).data,
                np.linspace(0, 100, num=101),
                axis=0,
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
            "constraints",
            np.percentile(
                torch.cat(self.ctx["constraints"], dim=0).data,
                np.linspace(0, 100, num=101),
                axis=0,
            ),
        )
        self.add_value(
            "reduced_constraints",
            np.percentile(
                torch.cat(self.ctx["reduced_constraints"], dim=0).data,
                np.linspace(0, 100, num=101),
                axis=0,
            ),
        )
        # I don't know whether I want to keep this, really
        # self.add_value("constraints_diagnostics", ...)
        if self.is_evaluation:
            self.add_value(
                "model_parameters",
                np.percentile(
                    self.ctx["model_parameters"][0].data,
                    np.linspace(0, 100, num=101),
                    axis=0,
                ),
            )
        else:
            # Percentile over the entire epoch and all parameters
            self.add_value(
                "model_parameters",
                np.percentile(
                    torch.stack(self.ctx["model_parameters"], dim=0).data,
                    np.linspace(0, 100, num=101),
                ),
            )
            self.add_value(
                "model_parameters_grad",
                np.percentile(
                    torch.stack(self.ctx["model_parameters_grad"], dim=0).data,
                    np.linspace(0, 100, num=101),
                ),
            )
        self.add_value(
            "timing",
            {
                key: np.average(
                    [
                        self.ctx["timing"][i][key]
                        for i in range(len(self.ctx["timing"]))
                    ],
                    weights=np.array(self.ctx["batch_size"]),
                )
                for key in self.ctx["timing"][0].keys()
            },
        )

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
        self.ctx["reduced_constraints"].append(
            self.get_tensor(engine.state.reduced_constraints)
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

