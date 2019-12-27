import sys

assert sys.version_info >= (3, 5)

import numpy as np

np.warnings.filterwarnings("ignore")
import argparse
import json
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


class S2CResult:
    def __init__(self, input_center = None, predict_heatmap = None, match=0, scale=[0,0,0], p_scan=None):
        """
        Simple class for representing result from the Scan2CAD network
        """
        self.input_center_file = input_center
        self.predict_heatmap_file = predict_heatmap
        self.match = match
        self.scale = scale
        self.p_scan = p_scan
        

def load_results(output_path="/data/scan2cad_output/full/"):
    """
    Analyze results from Scan2CAD network
    """
    all_outputs = [os.path.join(output_path, name) for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name)) and "testval_keypoint" in name ]
    results_list = []
    for subd in all_outputs:
        icf = os.path.join(subd, "input-center.vox")
        phf = os.path.join(subd, "predict-heatmap.vox2")
        pjson = os.path.join(subd, "predict.json")
        with open(pjson, "r") as predict_json:
            predict_result = json.load(predict_json)
            s2c_result = S2CResult()
            s2c_result.input_center_file = icf
            s2c_result.predict_heatmap_file = phf
            s2c_result.match = predict_result["match"]
            s2c_result.scale = predict_result["scale"]
            s2c_result.p_scan = predict_result["p_scan"]
            results_list.append(s2c_result)

    return results_list

if __name__ == "__main__":
    print("Analyze results from Scan2CAD network.")
    s2c_results = load_results()
    import pdb; pdb.set_trace()

