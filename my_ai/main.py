

import json
import logging

import multiprocessing

from src.core.my_ai import MyAI

logger = logging.getLogger(__name__)



if __name__ == "__main__":
    my_ai = MyAI()
    computer = multiprocessing.Process(target=my_ai.__run__, name="computer")
    computer.start()
    computer.join()