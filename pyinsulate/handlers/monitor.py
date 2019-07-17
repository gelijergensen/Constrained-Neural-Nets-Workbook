"""A handler which stores objects from an engine which are to be watched"""


class Monitor(object):
    """Monitors are run once at the end of an epoch and can be used to retrieve
    objects from the engine. Here we simply provide the tools for setting a new
    value onto the Monitor and for the retrieval of those values with the set()
    and get() methods."""

    def __init__(self):
        self._should_key_average = dict()
        self._counts = dict()

    def __call__(self, engine):
        raise NotImplementedError

    def add(self, key, average=False):
        if hasattr(self, key):
            print(f"Warning! Monitor already has key {key}")
        self._should_key_average[key] = average
        if average:
            setattr(self, key, None)
            self._counts[key] = 0
        else:
            setattr(self, key, list())

    def set(self, key, value):
        if not hasattr(self, key):
            raise AttributeError(
                f"Error! Monitor cannot set nonexistent attribute {key}"
            )
        if self._should_key_average[key]:
            if self._counts[key] == 0:
                setattr(self, key, value)
            else:
                old_value = self.get(key)
                new_value = (old_value * self._counts[key] + value) / \
                    (self._counts[key] + 1)
                setattr(self, key, new_value)
            self._counts[key] += 1
        else:
            getattr(self, key).append(value)

    def get(self, key):
        return getattr(self, key)
