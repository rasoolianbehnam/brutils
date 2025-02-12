import os
from pathlib import Path, PosixPath


class PathManager(PosixPath):
    def __init__(self, path):
        self.path = Path(path)

    @property
    def extension(self):
        return str(self).split(".")[-1]

    @property
    def dirname(self):
        return os.path.dirname(self)

    @property
    def basename(self):
        return os.path.basename(self)

    @property
    def remove_extension(self):
        return self.change_extension("")

    def change_extension(self, new_extension):
        out = str(self).split(".")[:-1] + [new_extension]
        return self.__class__(".".join(out))

    def append_to_name(self, prefix="", suffix=""):
        pass

    def change_dir(self, new_dir):
        return self.__class__(Path(new_dir) / Path(self.basename))
