"""Things to monitor during the training and evaluation phases"""

from ignite.engine import Events
import numpy as np
import torch

from src.handlers import Monitor


class TrainingMonitor(Monitor):
    def __init__(self, monitor_type="training"):
        super().__init__()
        self.monitor_type = monitor_type

        # epoch and iteration are recorded automatically
        self.add_key("batch_size")

        self.add_key("mean_loss")
        self.add_key("total_loss")
        self.add_key("constraints_error")
        self.add_key("constraints_percentiles")
        self.add_key("constraints_abs_percentiles")
        self.add_key("model_parameters_percentiles")
        if self.monitor_type == "training":
            self.add_key("model_parameters_grad_percentiles")

        self.add_key("timing")
        self.timing_keys = None

    def summarize(self):
        summary = f"Mean total loss: {self.total_loss[-1]:0.5f}, Mean data loss: {self.mean_loss[-1]:0.5f}, Mean constraint error: {self.constraints_error[-1]:0.5f}"
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
            "mean_loss",
            np.average(
                np.array(self.ctx["mean_loss"]),
                weights=np.array(self.ctx["batch_size"]),
            ),
        )
        if self.monitor_type != "projection":
            self.add_value(
                "total_loss",
                np.average(
                    np.array(self.ctx["total_loss"]),
                    weights=np.array(self.ctx["batch_size"]),
                ),
            )
        self.add_value(
            "constraints_error",
            np.average(
                np.array(self.ctx["constraints_error"]),
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
        # Percentile over the entire epoch and all parameters
        self.add_value(
            "model_parameters_percentiles",
            np.percentile(
                torch.stack(self.ctx["model_parameters"], dim=0).numpy(),
                np.linspace(0, 100, num=101),
            ),
        )
        if self.monitor_type == "training":
            self.add_value(
                "model_parameters_grad_percentiles",
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

        self.ctx["mean_loss"].append(
            self.get_tensor_item(engine.state.mean_loss)
        )
        if self.monitor_type != "projection":
            self.ctx["total_loss"].append(
                self.get_tensor_item(engine.state.total_loss)
            )
        self.ctx["constraints_error"].append(
            self.get_tensor_item(engine.state.constraints_error)
        )
        self.ctx["constraints"].append(
            self.get_tensor(engine.state.constraints)
        )
        self.ctx["model_parameters"].append(
            self.get_tensor(engine.state.model_parameters)
        )
        if self.monitor_type == "training":
            self.ctx["model_parameters_grad"].append(
                self.get_tensor(engine.state.model_parameters_grad)
            )

        self.ctx["timing"].append(engine.state.times.copy())
        if self.timing_keys is None:
            self.timing_keys = list(engine.state.times.keys())


class ProjectionMonitor(Monitor):
    def __init__(self):
        super().__init__()

        # epoch and iteration are recorded automatically
        self.add_key("batch_size")

        self.add_key("projection_epochs")
        self.add_key("mean_loss")
        self.add_key("constraints_error")
        self.add_key("constraints_percentiles")
        self.add_key("constraints_abs_percentiles")
        self.add_key("model_parameters_difference_percentiles")

        self.add_key("timing")
        self.timing_keys = None

    def summarize(self, during_projection=False):
        if during_projection:
            summary = f"Mean data loss: {self.ctx['epoch_mean_loss'][-1]:0.5f}, Mean constraint error: {self.ctx['epoch_constraints_error'][-1]:0.5f}"
        else:
            summary = f"Mean data loss: {self.mean_loss[-1][-1]:0.5f}, Mean constraint error: {self.constraints_error[-1][-1]:0.5f}"
        return summary

    @staticmethod
    def get_tensor(tensor):
        return tensor.cpu().detach()

    @staticmethod
    def get_tensor_item(tensor):
        return tensor.cpu().item()

    def should_stop_projection(self, tolerance):
        """This is a little outside of what this should actually do, but the 
        monitor does know what the current constraints error is"""
        return self.ctx["epoch_constraints_error"][-1] < tolerance

    def attach(self, engine):
        engine.add_event_handler(Events.STARTED, self.new_epoch)
        engine.add_event_handler(Events.EPOCH_STARTED, self.new_iteration)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.mark_epoch)
        engine.add_event_handler(Events.COMPLETED, self.end_epoch)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.__call__)

    def mark_epoch(self, engine):
        self.ctx["epoch_batch_size"].append(np.array(self.ctx["batch_size"]))
        self.ctx["epoch_mean_loss"].append(
            np.average(
                np.array(self.ctx["mean_loss"]),
                weights=np.array(self.ctx["batch_size"]),
            )
        )
        self.ctx["epoch_constraints_error"].append(
            np.average(
                np.array(self.ctx["constraints_error"]),
                weights=np.array(self.ctx["batch_size"]),
            )
        )
        self.ctx["epoch_constraints_percentiles"].append(
            np.percentile(
                torch.cat(self.ctx["constraints"], dim=0).numpy(),
                np.linspace(0, 100, num=101),
            )
        )
        self.ctx["epoch_constraints_abs_percentiles"].append(
            np.percentile(
                torch.abs(torch.cat(self.ctx["constraints"], dim=0)).numpy(),
                np.linspace(0, 100, num=101),
            )
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
        self.ctx["epoch_timing"].append(timing_dict)

        self.ctx["batch_size"] = list()
        self.ctx["mean_loss"] = list()
        self.ctx["constraints_error"] = list()
        self.ctx["timing"] = list()

    def finalize(self, engine):
        self.add_value("batch_size", np.array(self.ctx["epoch_batch_size"]))
        self.add_value(
            "projection_epochs", len(np.array(self.ctx["epoch_batch_size"]))
        )
        self.add_value("mean_loss", np.array(self.ctx["epoch_mean_loss"]))
        self.add_value(
            "constraints_error", np.array(self.ctx["epoch_constraints_error"])
        )
        self.add_value(
            "constraints_percentiles",
            np.array(self.ctx["epoch_constraints_percentiles"]),
        )
        self.add_value(
            "constraints_abs_percentiles",
            np.array(self.ctx["epoch_constraints_abs_percentiles"]),
        )
        # compute difference and percentiles thereof
        self.add_value(
            "model_parameters_difference_percentiles",
            np.percentile(
                (
                    self.ctx["model_parameters"][-1]
                    - self.ctx["model_parameters"][0]
                ).numpy(),
                np.linspace(0, 100, num=101),
            ),
        )

        self.add_value("timing", np.array(self.ctx["epoch_timing"]))

    def __call__(self, engine):
        self.ctx["batch_size"].append(len(self.get_tensor(engine.state.xb[0])))

        self.ctx["mean_loss"].append(
            self.get_tensor_item(engine.state.mean_loss)
        )
        self.ctx["constraints_error"].append(
            self.get_tensor_item(engine.state.constraints_error)
        )
        self.ctx["constraints"].append(
            self.get_tensor(engine.state.constraints)
        )
        self.ctx["model_parameters"].append(
            self.get_tensor(engine.state.model_parameters)
        )

        self.ctx["timing"].append(engine.state.times.copy())
        if self.timing_keys is None:
            self.timing_keys = list(engine.state.times.keys())
