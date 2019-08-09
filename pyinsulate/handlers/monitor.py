"""A handler which stores objects from an engine which are to be watched"""

from ignite.engine import Events


class Monitor(object):
    """Monitors tally objects per iteration and per epoch. They either record
    all instances of a value during the epoch or the average instance of the
    value during the epoch. Here we simply provide the tools for setting a new
    value onto the Monitor and for the retrieval of those values with the set()
    and get() methods."""

    def __init__(self):
        self._all_keys = list()
        self._should_key_average = dict()
        self._counts = dict()
        self._last_iteration = 0
        self._last_epoch = 0

        self.add("epoch", average=True)
        self.add("iteration", average=False)

    def __call__(self, engine):
        raise NotImplementedError

    def summarize(self):
        return "No Monitor Summary Given"

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self.new_epoch)
        engine.add_event_handler(Events.ITERATION_STARTED, self.new_iteration)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.__call__)

        def new_epoch(self, engine):
            super().new_epoch(engine)
            last_epoch = (
                self.get("epoch", -2) if len(self.get("epoch")) > 1 else 0
            )
            self.set("epoch", last_epoch + 1)

    def new_epoch(self, engine):
        for key in self._all_keys:
            if self._should_key_average[key]:
                self._counts[key] = 0
                getattr(self, key).append(None)
            else:
                getattr(self, key).append(list())
        # We have to do this this way because the evaluation engines always have epoch=1
        self.set("epoch", self._last_epoch + 1)
        self._last_epoch += 1

    def new_iteration(self, engine):
        self.set("iteration", self._last_iteration + 1)
        self._last_iteration += 1

    def add(self, key, average=False):
        if hasattr(self, key):
            print(f"Warning! Monitor already has key {key}")
        else:
            self._all_keys.append(key)
        setattr(self, key, list())
        self._should_key_average[key] = average

    def set(self, key, value):
        if not hasattr(self, key):
            raise AttributeError(
                f"Error! Monitor cannot set nonexistent attribute {key}"
            )
        if self._should_key_average[key]:
            if self._counts[key] == 0:
                getattr(self, key)[-1] = value
            else:
                old_value = self.get(key, -1)
                new_value = (old_value * self._counts[key] + value) / (
                    self._counts[key] + 1
                )
                getattr(self, key)[-1] = new_value
            self._counts[key] += 1
        else:
            getattr(self, key)[-1].append(value)

    def get(self, key, idxs=None):
        if idxs is None:
            return getattr(self, key)
        else:
            return getattr(self, key)[idxs]

    def keys(self):
        return iter(self._all_keys)

    def values(self):
        return iter(self.get(key) for key in self._all_keys)

    def items(self):
        return iter((key, self.get(key)) for key in self._all_keys)

    def __iter__(self):
        return self.keys()

    def __str__(self):
        return f"""{self.__class__.__name__}: {{{' '.join((f"'{key}'" for key in self._all_keys))}}}"""

    def __repr__(self):
        return f"{self.__class__}({repr(dict(self.items()))})"
