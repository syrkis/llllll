from chex import dataclass
from jax import random, tree_util, lax
from functools import partial
import darkdetect
import jax.numpy as jnp
from chex import dataclass
from dataclasses import field
import jaxmarl
from typing import Tuple, List, Dict, Optional, Callable
import parabellum as pb
import numpy as np

# from plot import int_to_color
# import utils
import datetime
from copy import deepcopy
import cv2
from time import time
import os
import pickle
import multiprocessing
from enum import Enum
from PIL import Image
import matplotlib.pyplot as plt

import parabellum as pb
from llllll import bts


# ### Objectives

# +

# ### Steps


# ### Parser


# +
class PlanParsingError(Exception):
    pass


def parse_list(txt):
    if txt[0] == "[" and txt[-1] == "]":
        if len(txt) == 2:
            return []
        for x in txt[1:-1].split(", "):
            if not x.isdigit():
                raise PlanParsingError(f'"{x}" is not a number.')
        return [int(x) for x in txt[1:-1].split(", ")]
    else:
        raise PlanParsingError(f'"{txt}" is not a list.')


def parse_coordinate(txt):
    if txt[0] == "(" and txt[-1] == ")":
        x, y = txt.split(", ")[0][1:], txt.split(", ")[1][:-1]
        if not x.isdigit():
            raise PlanParsingError(f'"{x}" is not a number.')
        if not y.isdigit():
            raise PlanParsingError(f'"{y}" is not a number.')
        return int(x), int(y)
    raise PlanParsingError(f'"{txt}" is not a tuple of integers.')
