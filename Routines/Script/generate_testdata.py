# The overall process of running benchmark evaluating performance of Scan2CAD
#
# 1. We generate "testset.json" with scenes in the validation split, using 
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
#
# Some potential issues:
# 1. Not sure on what model should I calculate 3D Harris keypoints. 
#

# load validation set

# load voxelized scene

# generate 3D Harris keypoints for each scene

# get the GT cads for each scene

# 
