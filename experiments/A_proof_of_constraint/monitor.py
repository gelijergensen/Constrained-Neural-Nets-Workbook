"""Things to monitor during the training and evaluation phases"""

import numpy as np

from pyinsulate.handlers import Monitor


class ProofOfConstraintMonitor(Monitor):
    def __init__(self):
        super().__init__()

        self.add("epoch", average=True)
        self.add("loss", average=False)
        self.add("mean_loss", average=False)
        self.add("constrained_loss", average=False)
        self.add("batch_size", average=False)
        self.add("constraints", average=False)
        self.add("reduced_constraints", average=False)
        self.add("constraints_diagnostics", average=False)
        self.add("timing", average=False)

    def new_epoch(self, engine):
        super().new_epoch(engine)
        last_epoch = self.get("epoch", -2) if len(self.get("epoch")) > 1 else 0
        self.set("epoch", last_epoch + 1)

    def summarize(self):
        mean_loss = np.mean(self.get("mean_loss", -1))
        mean_constrained_loss = np.mean(self.get("constrained_loss", -1))
        summary = f"Mean constrained loss: {mean_constrained_loss:0.5f}, Mean loss: {mean_loss:0.5f}"
        return summary

    def __call__(self, engine):
        self.set("loss", engine.state.loss.to("cpu"))
        self.set("mean_loss", engine.state.mean_loss.item())
        self.set("constrained_loss", engine.state.constrained_loss.item())
        self.set("batch_size", len(engine.state.xb))
        self.set("constraints", engine.state.constraints.to("cpu"))
        self.set("reduced_constraints", engine.state.reduced_constraints.item())
        self.set(
            "constraints_diagnostics",
            tuple(x.to("cpu") for x in engine.state.constraints_diagnostics),
        )
        self.set("timing", engine.state.times.copy())
