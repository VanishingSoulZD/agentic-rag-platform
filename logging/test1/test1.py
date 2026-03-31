import logging
from logging.handlers import RotatingFileHandler

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s| %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(
        "error.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file_handler)
    root.addHandler(error_handler)

    logger = logging.getLogger(__name__)
    logger.info("s")
    logger.warning("ss")
    logger.error("s")