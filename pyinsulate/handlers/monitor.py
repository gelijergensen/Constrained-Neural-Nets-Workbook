"""A handler which stores objects from an engine which are to be watched"""


class Monitor(object):

    def __init__(self):
        """Nothing needs to happen here"""

    def __call__(self, engine):
        raise NotImplementedError

    def set(self, key, value):
        getattr(self, key).append(value)

    def get(self, key):
        return getattr(self, key)
