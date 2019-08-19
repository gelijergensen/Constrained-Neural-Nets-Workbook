from torch import nn


class ProjectableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._current_mode = "train"
        self.train_state = None
        self.eval_state = None

    def train(self, mode="train"):
        if self.train_state is None:
            self.train_state = self.state_dict()
        if self.eval_state is None:
            self.eval_state = self.state_dict()
        super().train(mode != "eval")
        if self._current_mode == "train":
            self.train_state = self.state_dict()
        elif self._current_mode == "projection":
            self.eval_state = self.state_dict()

        if mode == "train":
            self.load_state_dict(self.train_state)
        elif mode == "eval":
            self.load_state_dict(self.eval_state)
        # regardless of what state we were in, if we move to projection, we keep the same parameters

        self._current_mode = mode

    def eval(self):
        self.train("eval")

    def proj(self):
        self.train("projection")
