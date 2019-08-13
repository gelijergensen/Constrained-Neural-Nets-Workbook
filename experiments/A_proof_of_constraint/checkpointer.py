"""Custom checkpointer to save out important configurations, all monitors, and
the model"""

from pyinsulate.handlers import Checkpointer


class ModelAndMonitorCheckpointer(Checkpointer):
    def __init__(
        self, dirname, filename_base, configuration, monitors, save_interval=1
    ):
        super().__init__(dirname, filename_base, save_interval)

        self.configuration = configuration
        self.monitors = monitors

    def retrieve(self, engine):
        # move the model and optimizer state dictionaries to cpu
        cpu_model_state_dict = engine.state.model_state_dict
        for k, v in cpu_model_state_dict.items():
            cpu_model_state_dict[k] = v.cpu()
        cpu_opt_state_dict = engine.state.optimizer_state_dict
        for k, v in cpu_opt_state_dict.items():
            cpu_opt_state_dict[k] = v.cpu()
        return {
            "epoch": engine.state.epoch,
            "configuration": self.configuration,
            "monitors": self.monitors,
            "model_state_dict": cpu_model_state_dict,
            "optimizer_state_dict": cpu_opt_state_dict,
        }
