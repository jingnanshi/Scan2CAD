import os
import json
import csv

from pathlib import Path

def load_scannet_split(dir_path, split_type="val"):
    """
    Load original splits for scannet
    """
    split = None
    if split_type == "val":
        split = []
        with open(os.path.join(dir_path, "scannetv2_val.txt")) as file:
            split = file.readlines()
        split = [x.strip() for x in split]

    elif split_type == "test":
        split = []
        with open(os.path.join(dir_path, "scannetv2_test.txt")) as file:
            split = file.readlines()
        split = [x.strip() for x in split]

    elif split_type == "train":
        split = []
        with open(os.path.join(dir_path, "scannetv2_train.txt")) as file:
            split = file.readlines()
        split = [x.strip() for x in split]

    return split

class S2C_QueryDB:
    def __init__(self, 
                 gtpath=os.path.join(str(Path.home()), "code/datasets/scan2cad_gt"),
                 shapenet_path=os.path.join(str(Path.home()),"code/datasets/ShapeNetCore.v2")):
        """
        Helper class for querying scan2cad ground truth annotation data
        """
        # load scan2cad gt data
        self.update_s2c_gt_db(gtpath)
        
    def update_s2c_gt_db(self, gtpath):
        """
        Load/update Scan2CAD ground truth database by reloading json/csv files
        """
        self.gt_db_path = gtpath
        self.gt_annotation_path = os.path.join(gtpath, "full_annotations.json")
        self.gt_unique_cads_path = os.path.join(gtpath, "unique_cads.csv")
        self.gt_cad_apperances_path = os.path.join(gtpath, "cad_appearances.json")
        
        # load full annotations
        self.gt_scene2aligned_models= {}
        self.gt_scene2trs = {}
        with open(self.gt_annotation_path) as json_file:
            data = json.load(json_file)
            for entry in data:
                scene_id = entry["id_scan"]
                scene_trs = entry["trs"]
                aligned_models = entry["aligned_models"]
                
                self.gt_scene2trs[scene_id] = scene_trs
                self.gt_scene2aligned_models[scene_id] = {}
                
                for model in aligned_models:
                    model_id = model["id_cad"]
                    model_cat = model["catid_cad"]
                    model_sym = model["sym"]
                    model_trs = model["trs"]
                    model_trs["center"] = model["center"]
                    model_keypoints_scan = model["keypoints_scan"]
                    model_keypoints_cad = model["keypoints_cad"]
                    
                    # a scene might have the same CAD aligned multiple times
                    if model_id not in self.gt_scene2aligned_models[scene_id]:
                        self.gt_scene2aligned_models[scene_id][model_id] = {
                            "catid_cad": model_cat,
                            "sym": model_sym,
                            "trs": [model_trs]
                        }
                    else:
                        self.gt_scene2aligned_models[scene_id][model_id]["trs"].append(model_trs)
                    
        # load unique cads from ground truth csv
        self.gt_unique_cads = set()
        self.gt_cad2synsets = {}
        self.gt_synset2cads = {} # find cad by synset
        with open(self.gt_unique_cads_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.gt_unique_cads.add(row['id-cad'])
                
                if row['id-cad'] not in self.gt_cad2synsets:
                    self.gt_cad2synsets[row['id-cad']] = [row['catid-cad']]
                else:
                    self.gt_cad2synsets[row['id-cad']].append(row['catid-cad'])
                
                if row['catid-cad'] not in self.gt_synset2cads:
                    self.gt_synset2cads[row['catid-cad']] = [row['id-cad']]
                else:
                    self.gt_synset2cads[row['catid-cad']].append(row['id-cad'])
    
        # load CADs per scene data
        self.gt_scene2cads = {}
        self.gt_scenes = set()
        with open(self.gt_cad_apperances_path) as json_file:
            data = json.load(json_file)
            for scene_id, cad_dict in data.items():
                self.gt_scene2cads[scene_id] = []
                self.gt_scenes.add(scene_id)
                for cad_id, count in cad_dict.items():
                    self.gt_scene2cads[scene_id].append((cad_id, count))
                # sort by count
                self.gt_scene2cads[scene_id].sort(key=lambda x:x[1])
            print("Loaded", len(self.gt_scenes), "scenes with ground truth data.")
            
            
    
    def get_gt_aligned_models_by_scene(self, scene):
        """
        Return a list of aligned models of provided scene in Scan2CAD 
        ground truth
        """
        return self.gt_scene2aligned_models[scene]
    
    def get_gt_cad_trs(self, scene, cad_id):
        """
        Return GT CAD model transformation
        
        Return:
        - a dictionary 
        { // <-- transformation from CAD space to world space 
        translation : [tx, ty, tz], // <-- translation vector
        rotation : [qw, qx, qy, qz], // <-- rotation quaternion
        scale : [sx, sy, sz] // <-- scale vector
        }
        - a symmetry string
        """
        return self.gt_scene2aligned_models[scene][cad_id]["trs"], self.gt_scene2aligned_models[scene][cad_id]["sym"]
    
    def get_gt_scene_trs(self, scene):
        """
        Return transformation from scan space to world space by scene
        
        Return:
        - a dictionary 
        { // <-- transformation from CAD space to world space 
        translation : [tx, ty, tz], // <-- translation vector
        rotation : [qw, qx, qy, qz], // <-- rotation quaternion
        scale : [sx, sy, sz] // <-- scale vector
        }
        """
        return self.gt_scene2trs[scene]
    
    def get_gt_scenes(self):
        """
        Return a list of scenes listed in the Scan2CAD ground truth
        """
        return self.gt_scenes
    
    def get_gt_cads_by_scene(self, scene):
        """
        Return a list of CAD models used in one scene
        
        Note: the returned CAD ids are in the format:
        synsetid_cadid
        """
        return self.gt_scene2cads[scene]
    
   
