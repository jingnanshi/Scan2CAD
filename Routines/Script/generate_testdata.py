import sys

assert sys.version_info >= (3, 5)

sys.path.append("../CropCentered")
import CropCentered

sys.path.append("../Keypoints2Grid")
import Keypoints2Grid

sys.path.append("../HarrisKeypoints")
import HarrisKeypoints

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

# global variables
s2c_db = utils.S2C_DB()
scannet_db = utils.ScanNet_DB()


def load_all_vox_cads(scenes):
    """
    Load all vox cads into a dictionary

    Return:
    - a dictionary with keys=synsetid_cadid, values=path_to_cad
    """
    s2c_db = utils.S2C_DB()
    scannet_db = utils.ScanNet_DB()

    loaded_cads = {}
    total_load_count = 0
    total_all_cads = 0
    for scene in scenes:
        gt_scene_cads = s2c_db.get_gt_cads_by_scene(scene)

        # load gt CADs
        load_count = 0
        for full_cad_id, count in gt_scene_cads:
            catid_cad, cad_id = full_cad_id.split("_")
            basename = "_".join(["valtest", full_cad_id])

            if full_cad_id not in loaded_cads:
                load_count += 1
                # load vox CAD
                voxfile_cad = (
                    params["shapenet_voxelized"]
                    + "/"
                    + catid_cad
                    + "/"
                    + cad_id
                    + "__0__.df"
                )

                # save CAD file
                filename_cad = Keypoints2Grid.save_empty_vox_cad(voxfile_cad, 
                        params["heatmaps"] + "/" + basename)
                loaded_cads[full_cad_id] = filename_cad

        total_load_count += load_count
        total_all_cads += len(gt_scene_cads)

    print("Actual load count:", total_load_count, "All CAD counts:", total_all_cads)

    return loaded_cads


def get_harris_keypoints_for_scene(scene, radius=0.3, nms_threshold=0.001, threads=8):
    """
    Get Harris keypoints for a scene
    """
    global scannet_db, s2c_db

    scene_path = scannet_db.get_scene_pc_path(scene)
    print("Scene path:", scene_path)
    scene_pc = o3d.io.read_point_cloud(scene_path)
    scene_pc_points = np.asarray(scene_pc.points)

    # generate 3D Harris keypoints for each scene
    harris_keypoints = HarrisKeypoints.get_harris_keypoints(
        scene_pc_points, radius, nms_threshold, threads
    )
    print("Harris shape:", harris_keypoints.shape)
    return harris_keypoints


def worker(scene, params, loaded_cads):
    """
    Parallized worker function
    """
    global scannet_db, s2c_db

    # prepare constants & variables
    voxfile_scan = (
        params["scannet_voxelized"]
        + "/"
        + scene 
        + "/"
        + scene 
        + "_res30mm_rot0.vox"
    )
    basename = "_".join([scene, "testval", "keypoint"]) + "_"

    # get harris keypoints
    harris_keypoints = get_harris_keypoints_for_scene(scene)

    # save all the scan crops
    kps_scan = np.transpose(harris_keypoints)
    kps_scan = kps_scan.reshape(3, -1, order="F")
    n_kps_scan = kps_scan.shape[1]
    kps_scan = np.asfortranarray(kps_scan)

    assert kps_scan.flags[
        "F_CONTIGUOUS"
    ], "Make sure keypoint array is col-major and continuous!"
    CropCentered.crop_and_save(
        63, -5 * 0.03, kps_scan, voxfile_scan, params["centers"] + "/" + basename
    )

    # get the GT cads for each scene
    # 1. iterate through all the CAD models
    # 2. for each cat model, iterate through all the scan keypoints
    test_data = []
    gt_scene_cads = s2c_db.get_gt_cads_by_scene(scene)
    for full_cad_id, count in gt_scene_cads:

        # iterate through all the Harris keypoints & save items
        for kp_idx in range(harris_keypoints.shape[0]):
            scale = [1, 1, 1]
            p_scan = kps_scan[0:3, kp_idx].tolist()
            filename_vox_center = (
                params["centers"] + "/" + basename + str(kp_idx) + ".vox"
            )
            filename_vox_heatmap = loaded_cads[full_cad_id]
            item = {
                "filename_vox_center": filename_vox_center,
                "filename_vox_heatmap": filename_vox_heatmap,
                "customname": basename,
                "p_scan": p_scan,
                "scale": scale,
                "match": True,
            }
            test_data.append(item)

    return test_data


if __name__ == "__main__":
    # Harris keypoints as p_scan points, and each GT cad as the heatmap CAD
    #
    # Each entry in json file:
    # 1. vox scene
    # 2. vox cad
    # 3. p_scan: harris point
    #
    # After feed forward through the neural network, for each entry in the json
    # there should be a corresponding output folder with 4 items;
    # 1. predict.json: match -- indicating whether its a match,
    #                  scale -- predicted scale diff in x,y,z
    #                  p_scan -- scan point (should be the same as the p_scan?)
    # 2. predict-heatmap.vox2: predicted heatmap by the network

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_type", action="store", default="val", dest="split_type"
    )
    parser.add_argument(
        "--params_file",
        action="store",
        default="./Parameters_for_valtest.json",
        dest="params",
    )
    parser_results = parser.parse_args()
    params = JSONHelper.read(parser_results.params)

    # load validation set
    all_scenes = utils.load_scannet_split(
        "../../Assets/scannet-metadata/", split_type=parser_results.split_type
    )

    # Generate test data
    loaded_cads = load_all_vox_cads(all_scenes)
    worker(all_scenes[10], params, loaded_cads)

    test_scenes = [all_scenes[0]]
    params_list = [params] * len(test_scenes)

    training_queue = queue.Queue()

    def task_done(future):
        try:
            result = future.result()  # blocks until results are ready
            training_queue.put(result)
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised %s" % error)

    # multiprocessing pools
    # data = []
    # pool = multiprocessing.Pool(processes=5)
    # data = pool.map(worker, zip(filtered_annotations, params_list))
    # with ProcessPool(max_workers=5) as pool:
    #    for i in zip(test_scenes, params_list):
    #        future = pool.schedule(worker, args=[i], timeout=300)
    #        future.add_done_callback(task_done)
