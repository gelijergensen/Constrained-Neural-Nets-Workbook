"""Custom checkpointer to save out important configurations, all monitors, and
the model"""

from src.handlers import Checkpointer


class ModelAndMonitorCheckpointer(Checkpointer):
    def __init__(
        self, dirname, filename_base, configuration, monitors, save_interval=1
    ):
        super().__init__(dirname, filename_base, save_interval)

        self.configuration = configuration
        self.monitors = monitors

    def retrieve(self, engine):
        return {
            "epoch": engine.state.epoch,
            "configuration": self.configuration,
            "monitors": self.monitors,
            "model_state_dict": engine.state.model_state_dict,
            "optimizer_state_dict": engine.state.optimizer_state_dict,
        }
