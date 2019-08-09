"""Things to monitor during the training and evaluation phases"""

import numpy as np

from pyinsulate.handlers import Monitor


class ProofOfConstraintMonitor(Monitor):
    def __init__(self):
        super().__init__()

        # epoch and iteration are recorded automatically
        self.add("loss", average=False)
        self.add("mean_loss", average=False)
        self.add("constrained_loss", average=False)
        self.add("batch_size", average=False)
        self.add("constraints", average=False)
        self.add("reduced_constraints", average=False)
        self.add("constraints_diagnostics", average=False)
        self.add("model_parameters", average=False)
        self.add("model_parameters_grad", average=False)
        self.add("timing", average=False)

    def summarize(self):
        mean_loss = np.mean(self.get("mean_loss", -1))
        mean_constrained_loss = np.mean(self.get("constrained_loss", -1))
        summary = f"Mean constrained loss: {mean_constrained_loss:0.5f}, Mean loss: {mean_loss:0.5f}"
        return summary

    def __call__(self, engine):
        self.set("loss", engine.state.loss.to("cpu").detach())
        self.set("mean_loss", engine.state.mean_loss.item())
        self.set("constrained_loss", engine.state.constrained_loss.item())
        self.set("batch_size", len(engine.state.xb))
        self.set("constraints", engine.state.constraints.to("cpu").detach())
        self.set("reduced_constraints", engine.state.reduced_constraints.item())
        self.set(
            "constraints_diagnostics",
            tuple(x.to("cpu") for x in engine.state.constraints_diagnostics),
        )
        self.set(
            "model_parameters", engine.state.model_parameters.to("cpu").detach()
        )
        grads = engine.state.model_parameters_grad
        if grads is not None:
            self.set("model_parameters_grad", grads.to("cpu").detach())
        else:
            self.set("model_parameters_grad", grads)
        self.set("timing", engine.state.times.copy())
