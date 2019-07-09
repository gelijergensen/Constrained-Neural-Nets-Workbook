"""A handler designed for use with Ignite which stores the value of some object,
possibly after applying some transformation"""

import traceback
import torch

from ignite.engine import Engine


class ObjectLogger(object):
    """A handler which grabs and stores some object, possibly with a 
    transformation"""

    def __init__(self, engine, retrieve_fn, clear_event=None):
        """Accepts the engine to run over and the name of the object to watch

        :param engine: an instance of ignite.engine.Engine
        :param retrieve_fn: function to retrieve the object with
        :param clear_event: optional event to register to clear the values list
        """
        if not isinstance(engine, Engine):
            raise TypeError("Argument engine should be an Engine")

        self.engine = engine
        self.retrieve_fn = retrieve_fn
        self.values = list()

        if clear_event is not None:
            self.engine.add_event_handler(clear_event, self.clear)

    def __call__(self, engine):
        try:
            self.values.append(self.retrieve_fn(engine))
        except Exception as e:
            print("Unable to retrieve object from the state using %s" %
                  self.retrieve_fn.__name__)
            traceback.print_exc()

    def clear(self):
        self.values = list()
