import os

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
