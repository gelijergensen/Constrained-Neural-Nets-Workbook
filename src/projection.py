from copy import deepcopy
from torch import nn


class ProjectableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._current_mode = "train"
        self.train_state = None
        self.proj_state = None
        self.eval_state = None
        self._is_dirty = False

    def train(self, mode="train"):
        if self.train_state is None:
            self.train_state = deepcopy(self.state_dict())
        if self.proj_state is None:
            self.proj_state = deepcopy(self.state_dict())
        if self.eval_state is None:
            self.eval_state = deepcopy(self.state_dict())
        super().train(mode != "eval")
        if self._current_mode == "train":
            self.train_state = deepcopy(self.state_dict())
            # also set this so we can evaluate
            self.eval_state = deepcopy(self.state_dict())
            self._is_dirty = True
        elif self._current_mode == "projection":
            self.proj_state = deepcopy(self.state_dict())
            # also set this so we can evaluate
            self.eval_state = deepcopy(self.state_dict())

        if mode == "train":
            self.load_state_dict(self.train_state)
        elif mode == "projection":
            if self._is_dirty:
                # we have trained since last projecting
                self.load_state_dict(self.train_state)
                self._is_dirty = False
            else:
                self.load_state_dict(self.proj_state)
        else:
            self.load_state_dict(self.eval_state)

        self._current_mode = mode

    def eval(self):
        self.train("eval")

    def proj(self):
        self.train("projection")
