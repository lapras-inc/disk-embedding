import logging
import os
import numpy as np

### Uncomment this line if you want to see the output at stdout as well.
# logging.basicConfig(level=logging.INFO)

def setup_logger(filename, also_stdout=False):
    current_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    logger = logging.getLogger(filename)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    fileHandler = logging.FileHandler(filename, mode="w")
    fileHandler.setFormatter(formatter)
    if also_stdout:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    if also_stdout:
        logger.addHandler(streamHandler)
    return logger


