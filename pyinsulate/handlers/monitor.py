"""A handler which stores objects from an engine which are to be watched"""

from ignite.engine import Events
import numpy as np


class _MonitoredObject(object):
    """An object which keeps numpy arrays of means, standard deviations, and 
    possibly also percentiles"""

    def __init__(self):
        self.mean = None
        self.std = None
        self.percentiles = None


class Monitor(object):
    """Monitors tally objects per iteration and per epoch. They either record
    all instances of a value during the epoch or the average instance of the
    value during the epoch. Here we simply provide the tools for setting a new
    value onto the Monitor and for the retrieval of those values with the set()
    and get() methods."""

    @staticmethod
    def _get_temp_attr_key(key):
        return f"_temporary_{key}_"

    def __init__(self):
        self._all_keys = list()
        self._record_mean = dict()
        self._record_std = dict()
        self._record_percentiles = dict()

        self._iterations_per_epoch = list()

    def __call__(self, engine):
        raise NotImplementedError

    def get_epochs(self):
        """Returns a 1d array with the epoch numbers"""
        return np.arange(1, len(self._iterations_per_epoch) + 1)

    def get_iterations(self):
        """Returns an array of arrays (possibly a 2d array) with the iteration
        numbers"""
        return np.array(
            [np.arange(num_iters) for num_iters in self._iterations_per_epoch]
        )

    def summarize(self):
        return "No Monitor Summary Given"

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self.new_epoch)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.end_epoch)
        engine.add_event_handler(Events.ITERATION_STARTED, self.new_iteration)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.__call__)

    def new_epoch(self, engine):
        for key in self._all_keys:
            # reset the temporary list
            setattr(self, self._get_temp_attr_key(key), list())
        self._iterations_per_epoch.append(0)

    def new_iteration(self, engine):
        self._iterations_per_epoch[-1] += 1

    def end_epoch(self, engine):
        for key in self._all_keys:
            # Extract the desired values
            value = np.array(getattr(self, self._get_temp_attr_key(key)))
            if self._record_mean[key]:
                prev = getattr(self, key).mean
                new = np.mean(value)
                if prev is None:
                    getattr(self, key).mean = new
                else:
                    getattr(self, key).mean = np.vstack((prev, new))
            if self._record_std[key]:
                prev = getattr(self, key).std
                new = np.std(value)
                if prev is None:
                    getattr(self, key).std = new
                else:
                    getattr(self, key).std = np.vstack((prev, new))
            if self._record_percentiles[key] is not None:
                prev = getattr(self, key).percentiles
                new = np.percentile(value, self._record_percentiles[key])
                if prev is None:
                    getattr(self, key).percentiles = new
                else:
                    getattr(self, key).percentiles = np.vstack((prev, new))

    def add_key(self, key, mean=False, std=False, percentiles=None):
        """Configure a new key to be monitored

        :param key: a string for the key name
        :param mean: whether to record the mean of the values
        :param std: whether to record the standard deviation of the values
        :param percentiles: a list of what percentiles to record. If not given,
            then no percentiles will be recorded
        """
        if hasattr(self, key):
            print(f"Warning! Monitor already has key {key}")
        else:
            if not mean and not std and percentiles is None:
                print(f"Warning! Monitor is not set to actually record {key}")
                print(f"Please specify mean, std, or percentiles")
            self._all_keys.append(key)
            self._record_mean[key] = mean
            self._record_std[key] = std
            self._record_percentiles[key] = percentiles
            setattr(self, key, _MonitoredObject())

    def add_value(self, key, value):
        """Adds a new value to the key

        :param key: key to add a new value for
        :param value: new value to be added
        """
        if key not in self._all_keys:
            raise AttributeError(
                f"Error! Monitor cannot set nonexistent attribute {key}"
            )
        # append the value to the temporary list
        getattr(self, self._get_temp_attr_key(key)).append(value)

    def __getattr__(self, key):
        # This catches the cases of trying to retrieve epoch/epochs or
        # iteration/iterations
        if key == "epochs" or key == "epoch":
            return self.get_epochs()
        elif key == "iterations" or key == "iteration":
            return self.get_iterations()
        else:
            raise AttributeError(f"Monitor does not have key {key}")

    def keys(self):
        """Returns an iterable over the keys of this monitor"""
        return iter(self._all_keys)

    def values(self):
        """Returns an iterable over the value of this monitor"""
        return iter(getattr(self, key) for key in self._all_keys)

    def items(self):
        """Returns an iterable over the key-value pairs of this monitor"""
        return iter((key, getattr(self, key)) for key in self._all_keys)

    def __iter__(self):
        return self.keys()

    def __str__(self):
        return f"""{self.__class__.__name__}: {{{' '.join((f"'{key}'" for key in self._all_keys))}}}"""

    def __repr__(self):
        return f"{self.__class__}({repr(dict(self.items()))})"
