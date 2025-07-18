from asyncio.log import logger
import json
from pathlib import Path
from typing import List

    


def split_list(input_list, chunk_size):
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]
