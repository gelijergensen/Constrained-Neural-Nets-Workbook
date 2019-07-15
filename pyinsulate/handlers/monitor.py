"""A handler which grabs various keys directly from an engine when called"""


class Monitor(object):

    def __init__(self, *keys):
        """Initializes an empty list for each key to watch"""
        self._keys = keys
        for key in self._keys:
            setattr(self, key, list())

    def __call__(self, engine):
        for key in self._keys:
            getattr(self, key).append(getattr(engine, key))

    def get(self, key):
        return getattr(self, key)
