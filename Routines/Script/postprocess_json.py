import json
from pathlib import Path


def filter_json(json_file_path, filtered_json_file_path):

    with open(json_file_path, "r") as json_file:
        train_data = json.load(json_file)
        filtered_train_data = []

        total_count = len(train_data)
        missing_files = 0
        for i in train_data:
            filename_vox_center = Path(i["filename_vox_center"])
            filename_vox_heatmap = Path(i["filename_vox_heatmap"])
            if filename_vox_heatmap.exists() and filename_vox_center.exists():
                filtered_train_data.append(i)
            else:
                missing_files += 1

        with open(filtered_json_file_path, "w") as f:
            json.dump(filtered_train_data, f)

        # dump to filtered trainset json
        print("For file:", json_file_path)
        print("Total pairs:", total_count)
        print("Filtered:", missing_files)


if __name__ == "__main__":
    trainset_path = "../../Assets/training-data/trainset.json"
    valset_path = "../../Assets/training-data/valset.json"
    filtered_trainset_path = "../../Assets/training-data/filtered_trainset.json"
    filtered_valset_path = "../../Assets/training-data/filtered_valset.json"

    # filter the json data files
    filter_json(trainset_path, filtered_trainset_path)
    filter_json(valset_path, filtered_valset_path)
