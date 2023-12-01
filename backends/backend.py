import os


class Backend:
    def load(self, path: str, **kargs):
        assert os.path.exists(path), f"File {path} does not exist."