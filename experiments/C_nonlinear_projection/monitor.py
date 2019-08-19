"""Things to monitor during the training and evaluation phases"""

import numpy as np
import torch

from pyinsulate.handlers import Monitor


class NonlinearProjectionMonitor(Monitor):
    def __init__(self, monitor_type="training"):
        super().__init__()
        self.monitor_type = monitor_type

        # epoch and iteration are recorded automatically
        self.add_key("batch_size")

        if self.monitor_type == "inference":
            self.inputs = None
            self.outputs = None
            self.recorded_inputs = False
            self.add_key("original_out")
            self.add_key("all_out")
            self.add_key("final_out")
            self.add_key("original_mean_loss")
            # self.add_key("all_mean_loss")
            self.add_key("final_mean_loss")
            self.add_key("original_constraints_error")
            # self.add_key("all_constraints_error")
            self.add_key("final_constraints_error")
            self.add_key("original_constraints_percentiles")
            # self.add_key("all_constraints_percentiles")
            self.add_key("final_constraints_percentiles")
            self.add_key("original_constraints_abs_percentiles")
            # self.add_key("all_constraints_abs_percentiles")
            self.add_key("final_constraints_abs_percentiles")
            self.add_key("model_parameter_differences_percentiles")
            self.add_key("projection_iterations")
        else:
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
        if self.monitor_type == "inference":
            summary = f"Original: (Loss: {self.original_mean_loss[-1]:0.5f}, Constraint error: {self.original_constraints_error[-1]:0.5f}). Final (average {np.average(self.projection_iterations[-1]):0.2f} iterations): (Loss: {self.final_mean_loss[-1]:0.5f}, Constraint error: {self.final_constraints_error[-1]:0.5f})"
        else:
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

        if self.monitor_type == "inference":
            if not self.recorded_inputs:
                self.inputs = torch.cat(self.ctx["inputs"], dim=0).numpy()
                self.outputs = torch.cat(self.ctx["outputs"], dim=0).numpy()
                self.recorded_inputs = True

            self.add_value(
                "original_out",
                torch.cat(self.ctx["original_out"], dim=0).numpy(),
            )
            self.add_value(
                "all_out",
                [
                    torch.cat(
                        [x[i] for x in self.ctx["all_out"]], dim=0
                    ).numpy()
                    for i in range(len(self.ctx["all_out"][0]))
                ],
            )
            self.add_value(
                "final_out", torch.cat(self.ctx["final_out"], dim=0).numpy()
            )
            self.add_value(
                "original_mean_loss",
                np.average(
                    np.array(self.ctx["original_mean_loss"]),
                    weights=np.array(self.ctx["batch_size"]),
                ),
            )
            self.add_value(
                "final_mean_loss",
                np.average(
                    np.array(self.ctx["final_mean_loss"]),
                    weights=np.array(self.ctx["batch_size"]),
                ),
            )
            self.add_value(
                "original_constraints_error",
                np.average(
                    np.array(self.ctx["original_constraints_error"]),
                    weights=np.array(self.ctx["batch_size"]),
                ),
            )
            self.add_value(
                "final_constraints_error",
                np.average(
                    np.array(self.ctx["final_constraints_error"]),
                    weights=np.array(self.ctx["batch_size"]),
                ),
            )
            self.add_value(
                "original_constraints_percentiles",
                np.percentile(
                    torch.cat(self.ctx["original_constraints"], dim=0).numpy(),
                    np.linspace(0, 100, num=101),
                ),
            )
            self.add_value(
                "final_constraints_percentiles",
                np.percentile(
                    torch.cat(self.ctx["final_constraints"], dim=0).numpy(),
                    np.linspace(0, 100, num=101),
                ),
            )
            self.add_value(
                "original_constraints_abs_percentiles",
                np.percentile(
                    torch.abs(
                        torch.cat(self.ctx["original_constraints"], dim=0)
                    ).numpy(),
                    np.linspace(0, 100, num=101),
                ),
            )
            self.add_value(
                "final_constraints_abs_percentiles",
                np.percentile(
                    torch.abs(
                        torch.cat(self.ctx["final_constraints"], dim=0)
                    ).numpy(),
                    np.linspace(0, 100, num=101),
                ),
            )
            # Percentile over the entire epoch and all parameters
            self.add_value(
                "model_parameter_differences_percentiles",
                np.percentile(
                    torch.stack(
                        self.ctx["model_parameter_differences"], dim=0
                    ).numpy(),
                    np.linspace(0, 100, num=101),
                ),
            )
            self.add_value(
                "projection_iterations", self.ctx["projection_iterations"]
            )
        else:
            self.add_value(
                "mean_loss",
                np.average(
                    np.array(self.ctx["mean_loss"]),
                    weights=np.array(self.ctx["batch_size"]),
                ),
            )
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
                    torch.abs(
                        torch.cat(self.ctx["constraints"], dim=0)
                    ).numpy(),
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

        if self.monitor_type == "inference":
            if not self.recorded_inputs:
                self.ctx["inputs"].append(self.get_tensor(engine.state.xb[0]))
                self.ctx["outputs"].append(self.get_tensor(engine.state.yb))
            self.ctx["original_out"].append(
                self.get_tensor(engine.state.original_out)
            )
            self.ctx["all_out"].append(
                [self.get_tensor(x) for x in engine.state.all_out]
            )
            self.ctx["final_out"].append(
                self.get_tensor(engine.state.final_out)
            )
            self.ctx["original_mean_loss"].append(
                self.get_tensor_item(engine.state.original_mean_loss)
            )
            self.ctx["final_mean_loss"].append(
                self.get_tensor_item(engine.state.final_mean_loss)
            )
            self.ctx["original_constraints_error"].append(
                self.get_tensor_item(engine.state.original_constraints_error)
            )
            self.ctx["final_constraints_error"].append(
                self.get_tensor_item(engine.state.final_constraints_error)
            )
            self.ctx["original_constraints"].append(
                self.get_tensor(engine.state.original_constraints)
            )
            self.ctx["final_constraints"].append(
                self.get_tensor(engine.state.final_constraints)
            )
            self.ctx["model_parameter_differences"].append(
                self.get_tensor(engine.state.model_parameter_differences)
            )
            self.ctx["projection_iterations"].append(
                engine.state.projection_iterations
            )
        else:
            self.ctx["mean_loss"].append(
                self.get_tensor_item(engine.state.mean_loss)
            )
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

