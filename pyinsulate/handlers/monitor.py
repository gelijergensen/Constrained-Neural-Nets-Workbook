"""A handler which stores objects from an engine which are to be watched"""


class Monitor(object):
    """Monitors are run once at the end of an epoch and can be used to retrieve
    objects from the engine. Here we simply provide the tools for setting a new
    value onto the Monitor and for the retrieval of those values with the set()
    and get() methods."""

    def __init__(self):
        pass  # Nothing needs to happen here for the base class

    def __call__(self, engine):
        raise NotImplementedError

    def set(self, key, value):
        if not hasattr(self, key):
            setattr(self, key, list())
        getattr(self, key).append(value)

    def get(self, key):
        return getattr(self, key)
