

import os
from tqdm import tqdm
import math
import torch
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
import text2pose.utils as utils


### INPUT
################################################################################

device = 'cpu'


### SETUP
################################################################################

# setup body model
body_model = BodyModel(bm_fname = config.SMPLH_NEUTRAL_BM, num_betas = config.n_betas)
body_model.eval()
body_model.to(device)

# load data
dataID_2_pose_info = utils.read_posescript_json("ids_2_dataset_sequence_and_frame_index.json")

# rotation transformation to apply so that the coordinates correspond to what we
# actually visualize (ie. from front view)
rotX = lambda theta: torch.tensor([[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]])

def transf(rotMat, theta_deg, values):
    theta_rad = math.pi * torch.tensor(theta_deg).float() / 180.0
    return rotMat(theta_rad).mm(values.t()).t()


### COMPUTE COORDINATES
################################################################################

coords = {}
coords_seq = {}
# compute all joint coordinates
test_number = 0
for dataID in tqdm(dataID_2_pose_info):

    test_number = test_number + 1
    if test_number > 5:
        break

    # load pose data
    pose_info = dataID_2_pose_info[dataID]
    pose = utils.get_pose_data_from_file(pose_info)
    pose_seq = utils.get_pose_sequence_data_from_file(pose_info)

    # infer coordinates
    with torch.no_grad():
        j = body_model(pose_body=pose[:,3:66], pose_hand=pose[:,66:], root_orient=pose[:,:3]).Jtr

        j_seq = body_model(pose_body=pose_seq[:, 3:66], pose_hand=pose_seq[:, 66:], root_orient=pose_seq[:, :3]).Jtr

    j = j.detach().cpu()[0]
    j = transf(rotX, -90, j)

    # import matplotlib.pyplot as plt
    # plt.scatter(j[:, 0], j[:, 1])
    # plt.show()
    # exit()




    j_seq = j_seq.detach().cpu()
    for frame_i in range(j_seq.shape[0]):
        j_seq[frame_i] = transf(rotX, -90, j_seq[frame_i])

    # store data
    coords[dataID] = j
    coords_seq[dataID] = j_seq
    break

# save poses
save_filepath = os.path.join(config.POSESCRIPT_LOCATION, "ids_2_coords_correct_orient_adapted.pt")
torch.save(coords, save_filepath)
print("Save coordinates at", save_filepath)

# save sequences
save_filepath = os.path.join(config.POSESCRIPT_LOCATION, "ids_2_coords_sequences_correct_orient_adapted.pt")
torch.save(coords, save_filepath)
print("Save coordinates at", save_filepath)

j = coords['0']
import matplotlib.pyplot as plt
plt.scatter(j[:, 0], j[:, 1])
plt.show()