import logging
import sys


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # don't log results for the non-master process
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s >> %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger