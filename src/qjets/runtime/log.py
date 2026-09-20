import logging
from pathlib import Path

def setup_logging(level: str, run_dir: Path) -> logging.Logger:
    log = logging.getLogger("qjets")
    log.setLevel(level)

    if log.hasHandlers():
        return log


    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "qjets.log")
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    return log
