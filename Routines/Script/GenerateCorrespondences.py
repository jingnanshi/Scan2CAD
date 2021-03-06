import sys

assert sys.version_info >= (3, 5)

sys.path.append("../CropCentered")
import CropCentered

sys.path.append("../Keypoints2Grid")
import Keypoints2Grid

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
import multiprocessing as mp
import CSVHelper
import JSONHelper
import utils
import time


def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M


def pick_random_vox_cad(params, not_cat_id):
    """
    Randomly pick a vox cad file from shapenet_voxelized directory
    excluding the not_cat_id file

    Return the path to the random vox shapenet cad file
    """
    shapenet_vox_path = params["shapenet_voxelized"]
    first_nodes = os.listdir(shapenet_vox_path)

    # select a different category
    cat_choice = random.choice(first_nodes)
    if not_cat_id is not None:
        while cat_choice == not_cat_id or not cat_choice.isdigit():
            cat_choice = random.choice(first_nodes)
    else:
        while not cat_choice.isdigit():
            cat_choice = random.choice(first_nodes)

    cad_choice_path = os.path.join(shapenet_vox_path, cat_choice)

    # select a vox cad file from this category
    second_nodes = os.listdir(cad_choice_path)
    cad_choice_file = random.choice(second_nodes)
    while cad_choice_file[-2:] != "df":
        cad_choice_file = random.choice(second_nodes)
    cad_id_choice = cad_choice_file.split("_")[0]

    voxfile_path = os.path.join(cad_choice_path, cad_choice_file)

    return cat_choice, cad_id_choice, voxfile_path


def gen_positive_aug_samples(r, params, num_per_cad_to_gen, training_data):
    """
    Generate positive (augmented data) for training.

    Randomly sample on CAD and use GT transform to the scan space

    Right now for each CAD model, we only sample one point
    """
    counter_cads = 0
    counter_heatmaps = 0
    id_scan = r["id_scan"]
    voxfile_scan = params["scannet_voxelized"] + "/" + id_scan + "/" + id_scan + "_res30mm_rot0.vox"
    Mscan = make_M_from_tqs(
        r["trs"]["translation"], r["trs"]["rotation"], r["trs"]["scale"]
    )

    for model in r["aligned_models"]:
        # the GT cat, cad id, gt_transform
        catid_cad = model["catid_cad"]
        id_cad = model["id_cad"]
        Mcad = make_M_from_tqs(
            model["trs"]["translation"], model["trs"]["rotation"], model["trs"]["scale"]
        )
        voxfile_cad = (
            params["shapenet_voxelized"] + "/" + catid_cad + "/" + id_cad + "__0__.df"
        )

        # Sample randmly on CAD surface
        for p_count in range(num_per_cad_to_gen):

            # basename for saving training data
            basename_trainingdata = "_".join(
                [id_scan, "aug", catid_cad, id_cad, str(counter_cads), str(p_count)]
            )

            while True:
                # sampled_kps_cad: in CAD frame
                sampled_kps_cad = Keypoints2Grid.get_random_cad_voxel_point(voxfile_cad)
                sampled_kps_cad = np.array(sampled_kps_cad).reshape(3, -1, order="F")
                assert sampled_kps_cad.shape[1] == 1
                sampled_kps_cad = np.vstack((sampled_kps_cad, np.ones((1, 1))))

                # use GT transform to transform it to world frame
                temp = np.dot(Mcad, sampled_kps_cad)

                # transform to scan frame
                sampled_kps_cad_scanframe = np.asfortranarray(
                    np.dot(np.linalg.inv(Mscan), temp)[0:3, :]
                )

                # check whether the point is close to scan surface
                close_to_surface = CropCentered.close_to_surface(
                    sampled_kps_cad_scanframe, voxfile_scan
                )

                if close_to_surface:
                    # save CAD heatmap
                    kps_cad = np.asfortranarray(sampled_kps_cad[0:3, :])
                    assert (
                        kps_cad.flags["F_CONTIGUOUS"] == True
                    ), "Make sure keypoint array is col-major and continuous!"

                    filename_vox_heatmap = Keypoints2Grid.single_project_and_save(
                        1.0,
                        kps_cad,
                        voxfile_cad,
                        params["heatmaps"] + "/" + basename_trainingdata,
                    )
                    if filename_vox_heatmap is None or len(filename_vox_heatmap) == 0:
                        print("WARNING: empty vox heatmap file name for", id_cad ,". Skip it.")
                        break 

                    # save centered crop
                    kps_scan = sampled_kps_cad_scanframe
                    assert kps_scan.flags[
                        "F_CONTIGUOUS"
                    ], "Make sure keypoint array is col-major and continuous!"
                    filename_vox_center = CropCentered.single_crop_and_save(
                        63,
                        -5 * 0.03,
                        kps_scan,
                        voxfile_scan,
                        params["centers"] + "/" + basename_trainingdata,
                    )
                    if filename_vox_center is None or len(filename_vox_center) == 0:
                        print("WARNING: empty vox heatmap file name. Resample.")
                        continue

                    # dump to file
                    scale = model["trs"]["scale"]
                    p_scan = kps_scan[0:3, 0].tolist()
                    item = {
                        "filename_vox_center": filename_vox_center,
                        "filename_vox_heatmap": filename_vox_heatmap,
                        "customname": basename_trainingdata,
                        "p_scan": p_scan,
                        "scale": scale,
                        "match": True,
                    }
                    training_data.append(item)
                    counter_heatmaps += 1
                    break
        counter_cads += 1

    print(
        "Generated positive augmented samples (heatmaps):",
        counter_heatmaps,
        "for",
        counter_cads,
        "cad models.",
    )
    return counter_cads, counter_heatmaps


def gen_negative_samples_1(r, params, num_to_gen, training_data):
    """
    Generate negative samples from randomly selected voxel point in the scan and a
    random CAD model
    """

    for counter in range(num_to_gen):
        # get scene voxel file path
        id_scan = r["id_scan"]
        Mscan = make_M_from_tqs(
            r["trs"]["translation"], r["trs"]["rotation"], r["trs"]["scale"]
        )
        voxfile_scan = (
            params["scannet_voxelized"] + "/" + id_scan + "/" + id_scan + "_res30mm_rot0.vox"
        )

        # basename for training data files
        basename_trainingdata = "_".join([id_scan, "neg1", str(counter)]) + "_"

        # randomly select voxel point in scene
        kps_scan_og = CropCentered.get_random_voxel_point(
            voxfile_scan
        )  # in world frame
        kps_scan = np.array(kps_scan_og).reshape(3, -1, order="F")
        n_kps_scan = kps_scan.shape[1]
        assert n_kps_scan == 1
        kps_scan = np.vstack((kps_scan, np.ones((1, n_kps_scan))))
        kps_scan = np.asfortranarray(np.dot(np.linalg.inv(Mscan), kps_scan)[0:3, :])
        assert kps_scan.flags[
            "F_CONTIGUOUS"
        ], "Make sure keypoint array is col-major and continuous!"
        filename_vox_center = CropCentered.single_crop_and_save(
            63,
            -5 * 0.03,
            kps_scan,
            voxfile_scan,
            params["centers"] + "/" + basename_trainingdata,
        )

        # randomly select cad model & gen heatmaps
        cat_choice, cad_id_choice, voxfile_cad = pick_random_vox_cad(params, None)
        filename_vox_heatmap = Keypoints2Grid.random_heatmap_and_save(
            voxfile_cad, params["heatmaps"] + "/" + basename_trainingdata
        )

        # save the training data json file
        scale = [random.random() * 2, random.random() * 2, random.random() * 2]
        p_scan = kps_scan[0:3, 0].tolist()
        filename_vox_center = params["centers"] + "/" + basename_trainingdata + ".vox"
        filename_vox_heatmap = (
            params["heatmaps"] + "/" + basename_trainingdata + ".vox2"
        )
        item = {
            "filename_vox_center": filename_vox_center,
            "filename_vox_heatmap": filename_vox_heatmap,
            "customname": basename_trainingdata,
            "p_scan": p_scan,
            "scale": scale,
            "match": False,
        }
        training_data.append(item)

    print("Generated negative samples of kind 1:", num_to_gen)
    return


def gen_negative_samples_2(r, params, training_data):
    """
    Generate negative samples from annoted keypoints and random CAD models
    """
    counter_cads = 0
    counter_heatmaps = 0

    for model in r["aligned_models"]:
        # the GT cat and cad id
        catid_cad = model["catid_cad"]
        id_cad = model["id_cad"]

        # get all annotated keypoints
        kps_scan_og = np.array(model["keypoints_scan"]["position"]).reshape(
            3, -1, order="F"
        )
        n_kps_scan_og = kps_scan_og.shape[1]

        # iterate through all the annotated keypoints
        # for each point, we pair it with a random CAD of the wrong class
        for i in range(n_kps_scan_og):
            # basename for training data
            basename_trainingdata = "_".join(
                [id_scan, "neg2", catid_cad, id_cad, str(counter_cads), str(i)]
            )
            p_scan = kps_scan_og[0:3, i].tolist()

            # generate centered crop (for the current scan keypoint only)
            kps_scan = np.array(p_scan).reshape(3, -1, order="F")
            n_kps_scan = kps_scan.shape[1]
            assert n_kps_scan == 1
            kps_scan = np.vstack((kps_scan, np.ones((1, n_kps_scan))))
            kps_scan = np.asfortranarray(np.dot(np.linalg.inv(Mscan), kps_scan)[0:3, :])
            assert kps_scan.flags[
                "F_CONTIGUOUS"
            ], "Make sure keypoint array is col-major and continuous!"
            filename_vox_center = CropCentered.single_crop_and_save(
                63,
                -5 * 0.03,
                kps_scan,
                voxfile_scan,
                params["centers"] + "/" + basename_trainingdata,
            )

            # Pick a random CAD model from a different class
            catid_cad, id_cad, voxfile_cad = pick_random_vox_cad(params, catid_cad)
            filename_vox_heatmap = Keypoints2Grid.random_heatmap_and_save(
                voxfile_cad, params["heatmaps"] + "/" + basename_trainingdata
            )

            # save the training data file
            scale = model["trs"]["scale"]
            item = {
                "filename_vox_center": filename_vox_center,
                "filename_vox_heatmap": filename_vox_heatmap,
                "customname": basename_trainingdata,
                "p_scan": p_scan,
                "scale": scale,
                "match": False,
            }
            training_data.append(item)
            counter_heatmaps += 1
        counter_cads += 1

    print(
        "Generated negative samples (heatmaps):",
        counter_heatmaps,
        "for",
        counter_cads,
        "cad models.",
    )
    return counter_cads, counter_heatmaps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_type", action="store", default="train", dest="split_type"
    )
    parser.add_argument(
        "--params_file", action="store", default="./Parameters.json", dest="params"
    )
    parser_results = parser.parse_args()
    params = JSONHelper.read(parser_results.params)

    # Load scannet split information
    scan_to_test = utils.load_scannet_split(
        params["scannet_metadata"], split_type=parser_results.split_type
    )

    print("NOTE: Symmetry not handled. You have to take care of it.")

    training_data = []
    scenes_left = len(scan_to_test)
    all_annotations = JSONHelper.read("./full_annotations.json")
    for r in all_annotations:
        id_scan = r["id_scan"]
        if id_scan not in scan_to_test:
            print("\nSkip ", id_scan)
            continue
        else:
            print("\nGenerate data for ", id_scan)

        voxfile_scan = (
            params["scannet_voxelized"] + "/" + id_scan + "/" + id_scan + "_res30mm_rot0.vox"
        )
        Mscan = make_M_from_tqs(
            r["trs"]["translation"], r["trs"]["rotation"], r["trs"]["scale"]
        )


        ###########################################################
        ## Positive training data: GT Annotations
        ###########################################################
        counter_cads = 0
        counter_heatmaps = 0
        for model in r["aligned_models"]:
            catid_cad = model["catid_cad"]
            id_cad = model["id_cad"]
            Mcad = make_M_from_tqs(
                model["trs"]["translation"],
                model["trs"]["rotation"],
                model["trs"]["scale"],
            )

            basename_trainingdata = (
                "_".join([id_scan, catid_cad, id_cad, str(counter_cads)]) + "_"
            )  # <-- this defines the basename of the training data for crops and heatmaps. pattern is "id_scan/catid_cad/id_cad/i_cad/i_kp"

            # -> Create CAD heatmaps
            voxfile_cad = (
                params["shapenet_voxelized"]
                + "/"
                + catid_cad
                + "/"
                + id_cad
                + "__0__.df"
            )
            kps_cad = np.array(model["keypoints_cad"]["position"]).reshape(
                3, -1, order="F"
            )
            n_kps_cad = kps_cad.shape[1]
            kps_cad = np.vstack((kps_cad, np.ones((1, n_kps_cad))))
            kps_cad = np.asfortranarray(np.dot(np.linalg.inv(Mcad), kps_cad)[0:3, :])
            # NOTE: Symmetry not handled. You have to take care of it.
            assert (
                kps_cad.flags["F_CONTIGUOUS"] == True
            ), "Make sure keypoint array is col-major and continuous!"
            Keypoints2Grid.project_and_save(
                1.0,
                kps_cad,
                voxfile_cad,
                params["heatmaps"] + "/" + basename_trainingdata,
            )
            # <-

            # -> Create scan centered crops
            kps_scan = np.array(model["keypoints_scan"]["position"]).reshape(
                3, -1, order="F"
            )
            n_kps_scan = kps_scan.shape[1]
            kps_scan = np.vstack((kps_scan, np.ones((1, n_kps_scan))))
            kps_scan = np.asfortranarray(np.dot(np.linalg.inv(Mscan), kps_scan)[0:3, :])
            assert kps_scan.flags[
                "F_CONTIGUOUS"
            ], "Make sure keypoint array is col-major and continuous!"
            CropCentered.crop_and_save(
                63,
                -5 * 0.03,
                kps_scan,
                voxfile_scan,
                params["centers"] + "/" + basename_trainingdata,
            )
            # <-

            # -> training list (to be read in by the network)
            scale = model["trs"]["scale"]
            for i in range(n_kps_scan):
                p_scan = kps_scan[0:3, i].tolist()
                filename_vox_center = (
                    params["centers"] + "/" + basename_trainingdata + str(i) + ".vox"
                )
                filename_vox_heatmap = (
                    params["heatmaps"] + "/" + basename_trainingdata + str(i) + ".vox2"
                )
                item = {
                    "filename_vox_center": filename_vox_center,
                    "filename_vox_heatmap": filename_vox_heatmap,
                    "customname": basename_trainingdata + str(i),
                    "p_scan": p_scan,
                    "scale": scale,
                    "match": True,
                }  # <-- in this demo only positive samples
                training_data.append(item)
                counter_heatmaps += 1
            counter_cads += 1
            # <-

        ###############################################
        ## Positive training data: Data augmentation ##
        ###############################################
        aug_data_num = int(counter_heatmaps * 10 / counter_cads)
        aug_counter_cads, aug_counter_heatmaps = gen_positive_aug_samples(
            r, params, aug_data_num, training_data
        )

        ###########################################
        ## Negative training data                ##
        ###########################################
        # Generate kind 2 first
        neg_counter_cads_2, neg_counter_heatmaps_2 = gen_negative_samples_2(
            r, params, training_data
        )
        # Generate kind 1 to make N_p to N_n 1:2
        neg_kind_1_num_to_gen = (
            2 * (aug_counter_heatmaps + counter_heatmaps) - neg_counter_heatmaps_2
        )
        gen_negative_samples_1(r, params, neg_kind_1_num_to_gen, training_data)

        print("\n********************************")
        print("For ", id_scan)
        print("\n********************************")
        print(
            "Generated positive training samples (heatmaps):",
            counter_heatmaps,
            "for",
            counter_cads,
            "cad models.",
        )
        print(
            "Generated augmented positive training samples (heatmaps):",
            aug_counter_heatmaps,
            "for",
            aug_counter_cads,
            "cad models.",
        )
        print(
            "Generated negative kind 1 training samples (heatmaps):",
            neg_kind_1_num_to_gen,
        )
        print(
            "Generated negative kind 2 training samples (heatmaps):",
            neg_counter_heatmaps_2,
            "for",
            neg_counter_cads_2,
            "cad models.",
        )
        print(
            "Overall: ",
            counter_heatmaps + aug_counter_heatmaps,
            " positive samples;",
            neg_kind_1_num_to_gen + neg_counter_heatmaps_2,
            " negative samples.",
        )
        print("\n********************************")
        scenes_left -= 1
        print("Scenes left:", scenes_left)
        print("\n********************************")

    if parser_results.split_type == "train":
        filename_json = "../../Assets/training-data/trainset.json"
    elif parser_results.split_type == "val":
        filename_json = "../../Assets/training-data/valset.json"
    else:
        filename_json = "../../Assets/training-data/unknownset.json"

    JSONHelper.write(filename_json, training_data)
    print("Training json-file (needed from network saved in:", filename_json)
