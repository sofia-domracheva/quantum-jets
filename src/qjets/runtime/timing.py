import time
from contextlib import contextmanager

class Timings:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()

        try:
            yield
        finally:
            self.stages[name] = time.perf_counter() - start

    def as_dict(self) -> dict[str, float]:
        return dict(self.stages)