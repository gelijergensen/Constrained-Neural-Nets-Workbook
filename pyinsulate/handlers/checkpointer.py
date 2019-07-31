"""Handler which accepts an object and saves to file periodically, taking care
to handle any errors that occur during saving. This is based on the 
ModelCheckpoint handler of ignite"""

from ignite.engine import Events
import os
import tempfile
import torch


class Checkpointer(object):
    """ObjectCheckpointers periodically save a particular object to file, given
    an implementation of the retrieve function which packages up the object to
    save."""

    def __init__(self, dirname, filename_base, save_interval=1):

        self._dirname = os.path.expanduser(dirname)
        self._filename_base = filename_base
        self.save_interval = save_interval
        self._iteration = 0

        os.makedirs(dirname, exist_ok=True)

        matched = [
            fname
            for fname in os.listdir(self._dirname)
            if fname.startswith(self._filename_base)
        ]
        if len(matched) > 0:
            raise ValueError(
                f"Files found matching {self._filename_base} in {self._dirname}. Cowardly refusing to construct new checkpointer and overwrite old files"
            )

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.__call__)

    def retrieve(self, engine):
        raise NotImplementedError

    def retrieve_and_save(self, engine):
        obj = self.retrieve(engine)
        filename = f"{self._filename_base}_{self._iteration:05d}.pth"
        path = os.path.join(self._dirname, filename)
        self._save(obj, path)

    def __call__(self, engine):
        self._iteration += 1
        if (self._iteration % self.save_interval) == 0:
            self.retrieve_and_save(engine)

    def _save(self, obj, path):
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=self._dirname)
        try:
            print(f"Saving checkpoint to {path}")
            torch.save(obj, tmp.file)
        except BaseException:
            print(f"Failed to save checkpoint!")
            tmp.close()
            os.remove(tmp.name)
            raise
        else:
            tmp.close()
            os.rename(tmp.name, path)
