import sys

assert sys.version_info >= (3, 5)

import numpy as np

np.warnings.filterwarnings("ignore")
import argparse
import glob
import re
import pathlib
import subprocess
import quaternion
import os
import shutil
import random
import CSVHelper
import JSONHelper
import utils
import time
import queue
import queue
from pebble import concurrent
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import open3d as o3d


def load_results(output_path):
    """
    Analyze results from Scan2CAD network
    """
    return

if __name__ == "__main__":
    print("Analyze results from Scan2CAD network.")
