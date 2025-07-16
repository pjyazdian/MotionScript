import json
import math
import os
import pickle
import random

import time
# $ streamlit run explore_posescript.py

import tqdm
from human_body_prior.body_model.body_model import BodyModel
import numpy as np
import sys

from text2pose.posescript.captioning_data import AUGMENTATION_LENGTH, AUGMENTATION_PORTION

local_path = "/localhome/pjomeyaz/Payam_Files/Projects/Gestcription/posescript/src/"
sys.path.append(local_path)

import text2pose.config as config
import text2pose.utils as utils
import text2pose.utils_visu as utils_visu
import text2pose.posescript.captioning as captioning_py
import warnings
warnings.filterwarnings('ignore')
import torch
import nltk


from MS_Algorithms import create_gif_with_blinking, merge_gifs_side_by_side

# get pose information
dataID_2_pose_info = utils.read_posescript_json("ids_2_dataset_sequence_and_frame_index.json")

# setup body model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
body_model = BodyModel(bm_fname=config.SMPLH_NEUTRAL_BM, num_betas=config.n_betas)
body_model.eval()
body_model.to(device)

# utils_visu.Visualize_anim(body_model, dataID_2_pose_info,
#                    dataID='607', out_folder='out_temp/captioning_motion',
#                    start_frame=200, end_frame=250,
#                    Skeleton3D_create=True, Mesh3D_create=True)

# utils_visu.Visualize_anim(body_model, dataID_2_pose_info,
#                    dataID='9000', out_folder='out_temp/captioning_motion',
#                    start_frame=250, end_frame=300,
#                    Skeleton3D_create=True, Mesh3D_create=True)


def get_pose_and_trans(pose_info, start_frame=None, end_frame=None,):
    pose_seq_data, trans = utils.get_pose_sequence_data_from_file(pose_info)

    start_frame= 0 if start_frame is None else start_frame
    end_frame = len(pose_seq_data) if end_frame is None else end_frame

    pose_seq_data = pose_seq_data[start_frame:end_frame] # Pose
    trans = trans[start_frame:end_frame] # Translation
    '''
    j_seq = body_model(pose_body=pose_seq_data[:, 3:66],
                       pose_hand=pose_seq_data[:, 66:],
                       root_orient=pose_seq_data[:, :3]).Jtr
    j_seq += trans[:, np.newaxis, :]
    j_seq = j_seq.float()
    # Transformation function:
    rotX = lambda theta: torch.tensor(
        [[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]])

    def transf(rotMat, theta_deg, values):
        theta_rad = math.pi * torch.tensor(theta_deg).float() / 180.0
        return rotMat(theta_rad).mm(values.t()).t()

    j_seq = j_seq.detach().cpu() #todo: fix rotation to just the first frame facing forward
    for frame_i in range(j_seq.shape[0]):
        j_seq[frame_i] = transf(rotX, -90, j_seq[frame_i])

    motions = j_seq
    # motions = motions.cpu().detach().numpy()
    '''
    motions = pose_seq_data

    return motions



rotX = lambda theta: torch.tensor(
            [[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]]).to(device)
rotY = lambda theta: torch.tensor(
            [[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]]).to(device)
rotZ = lambda theta: torch.tensor(
            [[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]]).to(device)

def transf(rotMat, theta_deg, values):
    theta_rad = math.pi * torch.tensor(theta_deg).float() / 180.0
    return rotMat(theta_rad).mm(values.t()).t()


def visualize_frames(pose_seq_data, trans, name):
    Mesh3D, Skeleton3D = False, True
    pose_seq_data =pose_seq_data.view(pose_seq_data.shape[0], -1)
    # 3D Mesh Visualization Animation
    if Mesh3D:
        print("Plotting 3D Mesh...")
        viewpoints = [[]]
        imgs = utils_visu.anim_from_pose_data(pose_seq_data, body_model, viewpoints=viewpoints, color="blue")
        for i, vp in enumerate(viewpoints):
            utils_visu.img2gif(imgs[i], f'out_temp/{name}_3DM_View({i}).gif')

    # 3D Skeleton Visualization
    if Skeleton3D:
        print("Plotting 3D skeleton...")

        # trans = torch.tensor(trans)
        # pose_seq_data = pose_seq_data.view(pose_seq_data.shape[0], -1, 3)
        # pose_seq_data += trans.unsqueeze(1)
        # pose_seq_data = pose_seq_data.view(pose_seq_data.shape[0], -1)
        if pose_seq_data.shape[1]>66:
            j_seq = body_model(pose_body=pose_seq_data[:, 3:66],
                               pose_hand=pose_seq_data[:, 66:],
                               root_orient=pose_seq_data[:, :3]).Jtr
        else:
            j_seq = body_model(pose_body=pose_seq_data[:, 3:66],
                               # pose_hand=pose_seq_data[:, 66:], HumanML3D has nof finger
                               root_orient=pose_seq_data[:, :3]).Jtr

        j_seq = j_seq.float()
        # Transformation function:


        j_seq = j_seq.detach().to(device)


        # Adding Translation
        # trans = torch.tensor(trans)
        # j_seq -= trans.unsqueeze(1)


        for frame_i in range(j_seq.shape[0]):


            trans_oriented  = torch.tensor(trans[frame_i]).unsqueeze(0).float().to(device)

            # 1. rotation around x-axis
            j_seq[frame_i] = transf(rotX, -90, j_seq[frame_i])


            # It is not clear why but seems rotation around Y axis should be
            # different for different samples to make sense [-90, 0, 90]
            trans_oriented = transf(rotX, -90, trans_oriented)
            trans_oriented = transf(rotY, -90, trans_oriented)
            # trans_oriented = transf(rotZ, -90, trans_oriented)

            j_seq[frame_i] += trans_oriented


        motions = j_seq
        motions = motions.cpu().detach().numpy()

        motion_batchified = np.expand_dims(motions, axis=0)
        title = ["Test"]
        # motion_batchified = torch.unsqueeze(motions, 0)
        utils_visu.draw_to_batch_Payam(motion_batchified, title, [f'out_temp/{name}_3DS.gif'])

# Shay
def visualize_frames_SFU_SALSA(pose_seq_data, trans, Motioncodes4vis, name, org_path, poses_rotvec):
    Mesh3D, Skeleton3D, Motioncodes, Merge = True, True, False, False

    pose_seq_data =pose_seq_data.view(pose_seq_data.shape[0], -1)
    # 3D Mesh Visualization Animation doesn't work for HumanML3D for now
    if Mesh3D:
        print("Plotting 3D Mesh...")
        viewpoints = [[]]

        imgs = utils_visu.anim_from_pose_data(torch.tensor(poses_rotvec[:, :66]).float().to(device), body_model, viewpoints=viewpoints, color="blue")
        for i, vp in enumerate(viewpoints):
            utils_visu.img2gif(imgs[i], f'out_temp/{name}/{name}_3D_Mesh.gif')

    # 3D Skeleton Visualization
    if Skeleton3D:
        print("Plotting 3D skeleton...")


        # if pose_seq_data.shape[1]>66:
        #     j_seq = body_model(pose_body=pose_seq_data[:, 3:66],
        #                        pose_hand=pose_seq_data[:, 66:],
        #                        root_orient=pose_seq_data[:, :3]).Jtr
        # else:
        #     j_seq = body_model(pose_body=pose_seq_data[:, 3:66],
        #                        # pose_hand=pose_seq_data[:, 66:], HumanML3D has nof finger
        #                        root_orient=pose_seq_data[:, :3]).Jtr

        # j_seq = j_seq.float()
        j_seq = pose_seq_data
        # Transformation function:


        j_seq = j_seq.detach().to(device)


        # Adding Translation
        # trans = torch.tensor(trans)
        # j_seq -= trans.unsqueeze(1)


        # for frame_i in range(j_seq.shape[0]):
        #     break
        #
        #     trans_oriented  = torch.tensor(trans[frame_i]).unsqueeze(0).float().to(device)
        #
        #     j_seq[frame_i] = transf(rotX, -90, j_seq[frame_i])
        #     trans_oriented = transf(rotX, -90, trans_oriented)
        #     j_seq[frame_i] += trans_oriented

        motions = j_seq
        motions = motions.cpu().detach().numpy()

        motion_batchified = np.expand_dims(motions, axis=0)
        title = [""]
        # motion_batchified = torch.unsqueeze(motions, 0)
        utils_visu.draw_to_batch_Payam(motion_batchified, title, [f'out_temp/{name}/{name}_3DS.gif'])


    if Motioncodes4vis == []:
        Motioncodes, Merge = False, False
    if Motioncodes:
        create_gif_with_blinking(Motioncodes4vis, total_frames=trans.shape[0], outname=f'out_temp/{name}/{name}_Motioncodes.gif')
    if Merge:
        merge_gifs_side_by_side(f'out_temp/{name}/{name}_3DS.gif',
                                f'out_temp/{name}/{name}_Motioncodes.gif',
                                f'out_temp/{name}/{name}_Merged.gif')

# rotX = lambda theta: torch.tensor([[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]]).to(device)

def transf(rotMat, theta_deg, values):
    theta_rad = math.pi * torch.tensor(theta_deg).float().to(device) / 180.0
    return rotMat(theta_rad).mm(values.t()).t()

def generate_text(file_index):
    import argparse
    from text2pose.config import POSESCRIPT_LOCATION

    parser = argparse.ArgumentParser(description='Parameters for captioning.')
    parser.add_argument('--action', default="generate_captions", choices=("generate_captions", "motioncode_stats"),
                        help="Action to perform.")
    parser.add_argument('--saving_dir', default='out_temp/captioning_motion' + "/generated_captions/",
                        help='General location for saving generated captions and data related to them.')
    parser.add_argument('--version_name', default="tmp",
                        help='Name of the caption version. Will be used to create a subdirectory of --saving_dir.')
    parser.add_argument('--simplified_captions', action='store_true',
                        help='Produce a simplified version of the captions (basically: no aggregation, no omitting of some support keypoints for the sake of flow, no randomly referring to a body part by a substitute word).')
    parser.add_argument('--apply_transrel_ripple_effect', action='store_true',
                        help='Discard some posecodes using ripple effect rules based on transitive relations between body parts.')
    parser.add_argument('--apply_stat_ripple_effect', action='store_true',
                        help='Discard some posecodes using ripple effect rules based on statistically frequent pairs and triplets of posecodes.')
    parser.add_argument('--random_skip', action='store_true', help='Randomly skip some non-essential posecodes.')
    parser.add_argument('--add_babel_info', action='store_true',
                        help='Add sentences using information extracted from BABEL.')
    parser.add_argument('--add_dancing_info', action='store_true',
                        help='Add a sentence stating that the pose is a dancing pose if it comes from DanceDB, provided that --add_babel_info is also set to True.')

    args = parser.parse_args()

    # create saving location
    save_dir = os.path.join(args.saving_dir, args.version_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print("Created new dir", save_dir)
    '''
    # load and format joint coordinates for input (dict -> matrix)
    # coords = torch.load(os.path.join(POSESCRIPT_LOCATION, "ids_2_coords_correct_orient_adapted.pt"))

    # </Payam> for test porpuse:
    # keys_to_grab = ['0', '1', '2', '3']
    # coords = {key: coords[key] for key in keys_to_grab if key in coords}
    # pose_info = dataID_2_pose_info['11982']
    # motions = get_pose_and_trans(pose_info, start_frame=250, end_frame=400, )

    # start_frame, end_frame = 3100, 3300 #S3 2352
    pose_info = dataID_2_pose_info['11982']
    start_frame, end_frame = 130, 1000 #230    # 1
    # start_frame, end_frame = 3200, 3500  # 2
    # start_frame, end_frame = 3100, 3300  # 3

    # pose_info = dataID_2_pose_info['11']
    # start_frame, end_frame = 130, 230  # 4

    # pose_info = dataID_2_pose_info['13']
    # start_frame, end_frame = 130, 430  # 5

    # pose_info = dataID_2_pose_info['6950']
    # start_frame, end_frame = 10000, 10400  # 6 *
    # start_frame, end_frame = 9000, 9100  # 7 *

    # start_frame, end_frame = 8000, 8300  # 8

    # start_frame += 100
    # end_frame += 220
    
    '''

    pose_info = dataID_2_pose_info[file_index]
    start_frame, end_frame = 1, 100  # 6 *

    pose_seq_data, trans = utils.get_pose_sequence_data_from_file(pose_info,
                                                                  normalizer_frame=start_frame)
    pose_seq_data = pose_seq_data.to(device)

    # infer coordinates
    with torch.no_grad():
        j_seq_3D_coord = body_model(pose_body=pose_seq_data[:, 3:66],
                           pose_hand=pose_seq_data[:, 66:],
                           root_orient=pose_seq_data[:, :3]).Jtr # U should decide about the root orentation that I want to use
    j_seq = j_seq_3D_coord.detach()
    for frame_i in range(j_seq.shape[0]):
        j_seq[frame_i] = transf(rotX, -90, j_seq[frame_i])

    "(Circular, 2260, 500, 700)"
    # ________________________________________________
    # 1.    coords [joints]
    coords = j_seq_3D_coord[start_frame:end_frame]

    # 2. Virtual joints
    # Adding two virtual joints {root_orientation and global_translation}
    # 2.1    coords [orientation-->rotvec]
    rotvec_seq = pose_seq_data[start_frame: end_frame]

    root_orient = rotvec_seq[:, :3].clone()
    deg2rad = lambda theta_deg: math.pi * theta_deg / 180.0
    rad2deg = lambda theta_rad: 180.0 * theta_rad / math.pi
    for frame in range(root_orient.shape[0]):
        theta_x, theta_y, theta_z = utils.rotvec_to_eulerangles(root_orient[frame, :].unsqueeze(0))
        root_orient[frame, :] = torch.cat((rad2deg(theta_x),
                                           rad2deg(theta_y),
                                           rad2deg(theta_z)))
    coords = torch.cat((coords, root_orient.unsqueeze(1)), dim=1)

    # 2.2    coords [translation]
    # Todo: we should rotate translation with respect to the transformation we use

    trans = torch.tensor(trans[start_frame:end_frame]).to(device)
    coords = torch.cat((coords, trans.unsqueeze(1)), dim=1)

    check_coord = coords.cpu().detach().numpy()
    # Check: _________________________________
    # trans = np.zeros_like(trans)
    visualize_frames(rotvec_seq, trans, file_index)
    exit()


    coords = coords.view(coords.shape[0], -1, 3)

    toplot = coords.cpu().detach().numpy()
    import matplotlib.pyplot as plt

    # for j in [20, 21]:
    #     for axis in [0, 1, 2]:
    #         plt.plot(coords[:,j, axis], label=f"Axis={axis}")
    #     plt.title(f"Joint={j}")
    #     plt.legend()
    #     plt.show()
    #     plt.clf()


    # Calculate the differences between consecutive elements



    for axis in [0, 1, 2]:
        plt.plot(root_orient.cpu().numpy()[:, axis], label=f"Axis={axis}")
    plt.title(f"Eular orientation")
    plt.legend()
    plt.show()
    plt.clf()

    for axis in [0, 1, 2]:
        plt.plot(trans.cpu().numpy()[:, axis], label=f"Axis={axis}")
    plt.title(f"trans")
    plt.legend()
    plt.show()
    plt.clf()
    # exit()
    #

    args.action = "generate_captions"
    if args.action == "generate_captions":

        pose_babel_text = False
        if args.add_babel_info:

            # get correspondences between tags and sentence parts
            babel_tag2txt_filepath = f"{os.path.dirname(os.path.realpath(__file__))}/action_to_sent_template.json"
            with open(babel_tag2txt_filepath, "r") as f:
                babel_tag2txt = json.load(f)

            # get a record of tags with no sentence correspondence
            null_tags = set([tag for tag in babel_tag2txt if not babel_tag2txt[tag]])

            # load and format babel labels for each pose
            pose_babel_tags_filepath = os.path.join(POSESCRIPT_LOCATION, "babel_labels_for_posescript.pkl")
            with open(pose_babel_tags_filepath, "rb") as f:
                d = pickle.load(f)
            pose_babel_tags = [d[pid] for pid in pose_ids]

            # filter out useless tags, and format results to have a list of
            # action tags (which can be empty) for each pose
            for i, pbt in enumerate(pose_babel_tags):
                if pbt is None or pbt == "__BMLhandball__":
                    pose_babel_tags[i] = []
                elif pbt == "__DanceDB__":
                    pose_babel_tags[i] = ["dance"] if args.add_dancing_info else []
                elif isinstance(pbt, list):
                    if len(pbt) == 0 or pbt[0][0] is None:
                        pose_babel_tags[i] = []
                    else:
                        # keep only action category labels
                        actions = []
                        for _, _, act_cat in pbt:
                            actions += act_cat
                        pose_babel_tags[i] = list(set(actions).difference(null_tags))
                else:
                    raise ValueError(str((i, pbt)))

            # create a sentence from BABEL tags for each pose, if available
            pose_babel_text = captioning_py.create_sentence_from_babel_tags(pose_babel_tags, babel_tag2txt)

        # process
        t1 = time.time()
        iters = 1
        # f_save = open(f'out_temp/{file_index}.txt', 'w')
        for cnt in tqdm.tqdm(range(iters)):
            binning_detial, motion_description =  captioning_py.main(coords,
                                                     save_dir=save_dir,
                                                     babel_info=pose_babel_text,
                                                     simplified_captions=args.simplified_captions,
                                                     apply_transrel_ripple_effect=args.apply_transrel_ripple_effect,
                                                     apply_stat_ripple_effect=args.apply_stat_ripple_effect,
                                                     random_skip=args.random_skip,
                                                     motion_tracking=True)
            if len(" ".join(motion_description)) < 10: return
            f_save = open(f'out_temp/{file_index}.txt', 'w')
            visualize_frames(rotvec_seq, trans, file_index)


            f_save.write(f'Trial #{cnt+1}:\n')
            f_save.write(binning_detial + '\n')
            import textwrap

            def wrap_by_words(text, num_words):
                words = text.split()
                lines = [' '.join(words[i:i + num_words]) for i in range(0, len(words), num_words)]
                return '\n'.join(lines)
            wrapped_motion_description = wrap_by_words(" ".join(motion_description), 15)  # Break lines at 10 words

            f_save.write("Description:\n" + wrapped_motion_description + '\n'
                         + ('_' * 113) + '\n' + ('*' * 113) + '\n' + ('_' * 113) +'\n'*5  )
        print(f"Description Generation + saving took {time.time() - t1} seconds for {iters} of iterations.")
        print(args)
        f_save.flush()
        f_save.close()

    if args.action == 'motioncode_stats':
        captioning_py.motioncode_stat_analysis(coords, save_dir)


def generate_text_HumanML3D(motion_id, motion_path, root_euler_path, start_frame=None, end_frame=None, motion_stats=False, ablations=[]):
    import argparse
    from text2pose.config import POSESCRIPT_LOCATION

    parser = argparse.ArgumentParser(description='Parameters for captioning.')
    parser.add_argument('--action', default="generate_captions", choices=("generate_captions", "motioncode_stats"),
                        help="Action to perform.")
    parser.add_argument('--saving_dir', default='out_temp/captioning_motion' + "/generated_captions/",
                        help='General location for saving generated captions and data related to them.')
    parser.add_argument('--version_name', default="tmp",
                        help='Name of the caption version. Will be used to create a subdirectory of --saving_dir.')
    parser.add_argument('--simplified_captions', action='store_true',
                        help='Produce a simplified version of the captions (basically: no aggregation, no omitting of some support keypoints for the sake of flow, no randomly referring to a body part by a substitute word).')
    parser.add_argument('--apply_transrel_ripple_effect', action='store_true',
                        help='Discard some posecodes using ripple effect rules based on transitive relations between body parts.')
    parser.add_argument('--apply_stat_ripple_effect', action='store_true',
                        help='Discard some posecodes using ripple effect rules based on statistically frequent pairs and triplets of posecodes.')
    parser.add_argument('--random_skip', action='store_true', help='Randomly skip some non-essential posecodes.')
    parser.add_argument('--add_babel_info', default=True, action='store_true',
                        help='Add sentences using information extracted from BABEL.')
    parser.add_argument('--add_dancing_info', action='store_true',
                        help='Add a sentence stating that the pose is a dancing pose if it comes from DanceDB, provided that --add_babel_info is also set to True.')

    args = parser.parse_args()

    # create saving location
    save_dir = os.path.join(args.saving_dir, args.version_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print("Created new dir", save_dir)

    # load and format joint coordinates for input (dict -> matrix)
    # coords = torch.load(os.path.join(POSESCRIPT_LOCATION, "ids_2_coords_correct_orient_adapted.pt"))


    # pose_info = dataID_2_pose_info[file_index]
    # start_frame, end_frame = 0, 40  # 6 *

    # Shay
    (FirstPerson_pose_seq_data, pose_seq_data,
     root_rot_vec, trans, poses_rotvec) = utils.get_pose_sequence_data_from_file_Salsa_Dance(motion_path,
                                                                                     root_euler_path,
                                                                                     normalizer_frame=start_frame)
    fps = 20.0
    # if len(pose_seq_data) < 40:
    #     return '', '', 0.0, 0.0
    SEQ_LENGTH = min(int(AUGMENTATION_LENGTH*fps), len(pose_seq_data))
    if len(pose_seq_data) < fps/2:
        return '', '', '', 0, 0
    start_frame = random.randint(0, len(pose_seq_data)-SEQ_LENGTH)  # since both included
    end_frame = start_frame + SEQ_LENGTH
    # start_frame, end_frame = int(2.4*20),  int(4.3*20 ) # len(pose_seq_data) # int(6.5*20 )# len(pose_seq_data) X#1
    start_frame, end_frame = 0, len(pose_seq_data)  # 40 # len(pose_seq_data) # For hamid

    start_time, end_time = float(start_frame) / fps, float(end_frame) / fps
    # XRZ
    FirstPerson_pose_seq_data = FirstPerson_pose_seq_data.to(device)
    pose_seq_data = pose_seq_data.to(device)
    root_rot_vec = torch.tensor(root_rot_vec).to(device)
    trans = torch.tensor(trans).to(device)


    if len(root_rot_vec) != len(pose_seq_data):
        root_rot_vec = root_rot_vec[:len(pose_seq_data)]
    FirstPerson_pose_seq_data = FirstPerson_pose_seq_data[start_frame:end_frame]
    pose_seq_data = pose_seq_data[start_frame:end_frame]
    root_rot_vec = root_rot_vec[start_frame:end_frame]
    trans = trans[start_frame:end_frame]


    # pose_seq_data = pose_seq_data[:end_frame] #todo: REMOVE th
    # pose_seq_data_shape = pose_seq_data.shape
    # infer coordinates
    # with torch.no_grad():
    #     j_seq_3D_coord = body_model(pose_body=pose_seq_data[:, 3:66],
    #                                 # pose_hand=pose_seq_data[:, 66:],
    #                                 root_orient=pose_seq_data[:,:3]).Jtr  # U should decide about the root orentation that I want to use

    FirstPerson_pose_seq_data = FirstPerson_pose_seq_data.reshape(pose_seq_data.shape[0], -1, 3)
    pose_seq_data = pose_seq_data.reshape(pose_seq_data.shape[0], -1, 3)
    # for frame_i in range(pose_seq_data.shape[0]):
    #
    #
    #     trans_oriented  = torch.tensor(trans[frame_i]).unsqueeze(0).float().to(device)
    #
    #     pose_seq_data[frame_i] = transf(rotX, -180, pose_seq_data[frame_i])
    #     trans_oriented = transf(rotX, -180, trans_oriented)
    #     pose_seq_data[frame_i] += trans_oriented
    #
    # pose_seq_data = (pose_seq_data).reshape(pose_seq_data_shape)

    # j_seq_3D_coord = pose_seq_data.view(pose_seq_data.shape[0], -1, 3)
    # j_seq = j_seq_3D_coord.detach()
    # # for frame_i in range(j_seq.shape[0]):
    # #     j_seq[frame_i] = transf(rotX, -90, j_seq[frame_i])
    j_seq = pose_seq_data
    # "(Circular, 2260, 500, 700)"
    # ________________________________________________
    # 1.    coords [joints]

    def normalize_coords(pose_seq_data_x, trans_x):
        for frame in range(pose_seq_data_x.shape[0]):
            for j_id in range(pose_seq_data_x.shape[1]):
                pose_seq_data_x[frame, j_id] -= trans_x[frame]
        return pose_seq_data_x

    # coords = pose_seq_data
    # # coords = normalize_coords(pose_seq_data, trans)
    coords = FirstPerson_pose_seq_data



    # 2. Virtual joints
    # Adding two virtual joints {root_orientation and global_translation}
    # 2.1    coords [orientation-->rotvec]
    # rotvec_seq = pose_seq_data[start_frame: end_frame]

    # Eroor: joint number zero is pelvis, not root orientation
    # root_orient = pose_seq_data[start_frame:end_frame, 0, :3].clone()
    # root_orient = root_rot_eular[start_frame:end_frame]
    root_euler_orient = torch.zeros_like(root_rot_vec).to(device)
    # deg2rad = lambda theta_deg: math.pi * theta_deg / 180.0
    rad2deg = lambda theta_rad: 180.0 * theta_rad / math.pi


    for frame in range(root_euler_orient.shape[0]):

        # theta_x, theta_y, theta_z = root_orient[frame, [0, 1, 2]] # utils.rotvec_to_eulerangles(root_orient[frame, :].unsqueeze(0))

        theta_x, theta_y, theta_z = utils.rotvec_to_eulerangles(root_rot_vec[frame, :].unsqueeze(0))
        # root_orient[frame, :] = torch.cat((rad2deg(theta_x),
        #                                    rad2deg(theta_y),
        #                                    rad2deg(theta_z)))


        root_euler_orient[frame, :] = torch.cat((  torch.unsqueeze(rad2deg(theta_x), 0),
                                                   torch.unsqueeze(rad2deg(theta_y), 0),
                                                   torch.unsqueeze(rad2deg(theta_z), 0))).squeeze()

    # To adjust the orientation w.r.t. mirrored samples (swap left/right)
    def adjust_mirroed_root_orientations(batch_orientations):
        # Negate the yaw and roll for the mirrored pose
        batch_orientations[:, 1] = -batch_orientations[:, 1]  # Roll
        batch_orientations[:, 2] = -batch_orientations[:, 2]  # Yaw
        return batch_orientations

    # To normalize w.r.t. the first frame facing forwadd.
    def normalize_angles(angles):
        # Normalize angles to range [0, 360)
        angles = angles % 360
        # Shift angles > 180 to [-180, 180)
        angles[angles > 180] -= 360
        return angles

    if 'M' in motion_id:
        root_euler_orient = adjust_mirroed_root_orientations(root_euler_orient)

    angle_diff = root_euler_orient[0, 2]
    root_euler_orient[:, 2] = normalize_angles((root_euler_orient[:, 2] - angle_diff))  # not sure if this is correct for negatives

    # In order to be compatible with motionscript we add zeros for fingers
    # Later, they would be ignored in the prepare_input() function
    # coords = torch.cat((coords, torch.zeros(coords.shape[0], 2*15, 3).to(coords.device)), dim=1)
    coords = coords[:, :52,: ]

    coords = torch.cat((coords, root_euler_orient.unsqueeze(1)), dim=1)


    # 2.2    coords [translation]
    # Todo: we should rotate translation with respect to the transformation we use
    # trans = trans[start_frame:end_frame]
    coords = torch.cat((coords, trans.unsqueeze(1)), dim=1)

    check_coord = coords.cpu().detach().numpy()
    # Check: _________________________________
    # trans = np.zeros_like(trans)
    # visualize_frames_HumanML3D(pose_seq_data, trans, [], motion_id, '', poses_rotvec)
    # (pose_seq_data, trans, Motioncodes4vis, name, org_path)
    # exit()

    coords = coords.view(coords.shape[0], -1, 3)

    toplot = coords.cpu().detach().numpy()
    import matplotlib.pyplot as plt

    # for j in [53]:
    #     for axis in [0, 1, 2]:
    #         plt.plot(coords.cpu().detach().numpy()[:,j, axis], label=f"Axis={axis}")
    #     plt.title(f"Joint={captioning_py.swapped_JOINT_NAMES2ID[j]}")
    #     plt.legend()
    #     plt.show()
    #     plt.clf()
    # exit()
    # Calculate the differences between consecutive elements
    if True:
        root_orient2print = root_euler_orient.cpu().numpy()
        for axis in [0, 1, 2]:
            plt.plot(root_orient2print[:, axis], label=f"Axis={axis}")
        plt.title(f"Eular orientation")
        plt.legend()
        plt.show()
        # plt.clf()

        for axis in [0, 1, 2]:
            plt.plot(trans[:, axis].cpu().numpy(), label=f"Axis={axis}")
        plt.title(f"trans")
        plt.legend()
        plt.show()
        plt.clf()
        # exit()




    args.action = "generate_captions" if motion_stats==False else "motioncode_stats"
    if args.action == "generate_captions":

        pose_babel_text = False
        args.add_babel_info = True
        if args.add_babel_info:

            # get correspondences between tags and sentence parts
            babel_tag2txt_filepath = f"{os.path.dirname(os.path.realpath(__file__))}/action_to_sent_template.json"
            with open(babel_tag2txt_filepath, "r") as f:
                babel_tag2txt = json.load(f)

            # get a record of tags with no sentence correspondence
            null_tags = set([tag for tag in babel_tag2txt if not babel_tag2txt[tag]])

            # load and format babel labels for each motion w.r.t. to the selected time window
            HML3D_babel_tags_filepath = os.path.join(config.POSESCRIPT_LOCATION, "babel_labels_for_motionscript.pkl")
            # pose_babel_tags_filepath = os.path.join(POSESCRIPT_LOCATION, "babel_labels_for_posescript.pkl")
            with open(HML3D_babel_tags_filepath, "rb") as f:
                HML3D_BABEL = pickle.load(f)
            '''
            pose_babel_tags = [d[pid] for pid in pose_ids]

            # # filter out useless tags, and format results to have a list of
            # # action tags (which can be empty) for each pose
            # for i, pbt in enumerate(pose_babel_tags):
            #     if pbt is None or pbt == "__BMLhandball__":
            #         pose_babel_tags[i] = []
            #     elif pbt == "__DanceDB__":
            #         pose_babel_tags[i] = ["dance"] if args.add_dancing_info else []
            #     elif isinstance(pbt, list):
            #         if len(pbt) == 0 or pbt[0][0] is None:
            #             pose_babel_tags[i] = []
            #         else:
            #             # keep only action category labels
            #             actions = []
            #             for _, _, act_cat in pbt:
            #                 actions += act_cat
            #             pose_babel_tags[i] = list(set(actions).difference(null_tags))
            #     else:
            #         raise ValueError(str((i, pbt)))

            # create a sentence from BABEL tags for each pose, if available
            pose_babel_text = captioning_py.create_sentence_from_babel_tags(pose_babel_tags, babel_tag2txt)
            '''
            motion_babel_text, motion_babel_details = captioning_py.create_sentence_from_babel_to_hml3d(HML3D_BABEL, babel_tag2txt,
                                                                                  motion_id,
                                                                                  start_time,
                                                                                  end_time,
                                                                                  GPT_Template='BABEL') # GPT_Template='GPT')

        # process
        else:
            motion_babel_text, motion_babel_details = '', ''

        Ablation_No_motionScript = False
        if Ablation_No_motionScript:
            binning_detial, motioncodes4vis, motion_descriptions_non_agg, motion_description = '', [''], ['']
        else:
            binning_detial, motioncodes4vis, motion_descriptions_non_agg, motion_description = captioning_py.main(coords,
                                                                    save_dir=save_dir,
                                                                    babel_info=pose_babel_text,
                                                                    simplified_captions=args.simplified_captions,
                                                                    apply_transrel_ripple_effect=args.apply_transrel_ripple_effect,
                                                                    apply_stat_ripple_effect=args.apply_stat_ripple_effect,
                                                                    random_skip=args.random_skip,
                                                                    motion_tracking=True, ablations=ablations)
            # Shay
            visualize_frames_SFU_SALSA(pose_seq_data, trans, motioncodes4vis, motion_id, motion_path, poses_rotvec)
        if ' '.join(motion_description).strip() == '':
            binning_detial, motion_descriptions_non_agg, motion_description = '', [''], ['']
        else:
            motion_descriptions_non_agg = [motion_babel_text.strip()] + motion_descriptions_non_agg
            motion_description = [motion_babel_text.strip()] + motion_description
        return ((binning_detial + str(2*"\n") + str(10*" ") + " BABEL Captions:\n\n *W.R.T. HumanML3D frames from AMASS\n" + motion_babel_details),
                " ".join([x for x in motion_descriptions_non_agg if x!='']),
                " ".join([x for x in motion_description if x!='']),
                start_time, end_time)

    if args.action == 'motioncode_stats':
        # captioning_py.motioncode_stat_analysis(coords, save_dir)
        return captioning_py.motioncode_stat_analysis_step1_extraction(coords, save_dir)



#3292   4676
# for file_index in tqdm.tqdm(range(3292, 20000)):
#     # if dataID_2_pose_info[str(file_index)][0] not in ['000531']: continue
#     generate_text(str(file_index))
#     break
# exit()
humanml3d_path = '..\\..\\..\\data\\HumanML3D'

ids_file = open(f'{humanml3d_path}\\all.txt', 'r')
counter_line, counter_gen, len_array, problem_list = 0, 0, [], []
generated_label = ''
lines= ids_file.read().split('\n')

# import os
# word_counts = []
# folder_path = f'out_temp\\motion_script_only_\\'
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):  # Check if it's a text file
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, 'r', encoding='utf-8') as file:
#             text = file.read()
#             caption = text.split('#')[0]
#         word_count = (len(nltk.word_tokenize(caption)))
#         word_counts.append(word_count)
#         # print(f'File: {filename}, Word Count: {word_count}')

# Calculating average and standard deviation
# average_word_count = np.mean(word_counts)
# # std_dev_word_count = np.std(word_counts)
# print(f'Average Word Count: {average_word_count}')
# print(f'Standard Deviation of Word Count: {std_dev_word_count}')
# exit()




counter = 0
# # Save the current file descriptor
original_stdout = sys.stdout
# # Redirect stdout to devnull to suppress print statements
# sys.stdout = open(os.devnull, 'w')


# Redirect stdout to a string buffer
import io
sys.stdout = io.StringIO()


# all_motion_stats = pickle.load(open(os.path.join('out_temp/statistics/', 'all_motion_stats_step1.pkl'), 'rb'))
# captioning_py.motioncode_stat_analysis_step2_visualization(all_motion_stats, 'out_temp/statistics/')
# exit()


motion_stats = False
all_motion_stats = None

# ['intensity', 'velocity', 'chronological']
# ablation_list = ['intensity']

for sample_index in tqdm.tqdm(range(0*len(lines)//5, 2*len(lines)//5 )):
    line = lines[sample_index]
    id = line.rstrip('\n')

    print(f'ID ----> {id}')



    counter += 1
    # id = '000406'
    # # if counter < 4842: continue
    # # if counter > 5000: continue

    # Load paths
    motion_path = f'{humanml3d_path}\\new_joint_vecs\\{id}.npy'
    euler_oath = f'{humanml3d_path}\\new_joint_vecs\\{id}.npy'
    org_caption = open(f'{humanml3d_path}\\texts\\{id}.txt', 'r', encoding='utf-8').read()

    motion_path = 'M:\\Pair1_8_7_take1_1_subject-Pair1_8_7_follower_subject_stageii.npz'


    save_path_text_sanity = f'out_temp\\sanity_check\\{id}.txt'
    if not os.path.exists('out_temp\\sanity_check'):
        os.makedirs('out_temp\\sanity_check')
    save_path_text_ms = f'out_temp\\motion_script_only\\{id}.txt'
    if not os.path.exists('out_temp\\motion_script_only'):
        os.makedirs('out_temp\\motion_script_only')
    save_path_text_mshml3d = f'out_temp\\combined\\{id}.txt'
    if not os.path.exists('out_temp\\combined'):
        os.makedirs('out_temp\\combined')

    # save_path_text_sanity = f'out_temp\\sanity_check_\\{id}.txt'
    # save_path_text_ms = f'out_temp\\motion_script_only_\\{id}.txt'
    # save_path_text_mshml3d = f'out_temp\\combined_\\{id}.txt'


    # # Temp for testing:
    save_path_sample_id = f'out_temp\\{id}'
    if not os.path.exists(save_path_sample_id):
        os.makedirs(save_path_sample_id)
    save_path_text_sanity = f'{save_path_sample_id}\\sanity_check_{id}.txt'
    save_path_text_ms = f'{save_path_sample_id}\\MS_only_{id}.txt'
    save_path_text_mshml3d = f'{save_path_sample_id}\\combined_{id}.txt'


    # if os.path.exists(save_path_text_ms): continue
    # print("id", id)

    if os.path.exists(save_path_text_sanity): continue
    root_euler_path = f'{humanml3d_path}\\root_euler\\{id}.npy'

    if True:



        if motion_stats:
            current_intptt_stats = generate_text_HumanML3D(id, motion_path, root_euler_path,
                                                             motion_stats=motion_stats)
            if all_motion_stats is None:
                all_motion_stats = current_intptt_stats
            else:
                for m_kind in all_motion_stats:
                    for j_id in range(len(all_motion_stats[m_kind])):
                        try:
                            all_motion_stats[m_kind][j_id].extend(current_intptt_stats[m_kind][j_id])
                        except:
                            print()
            continue


        def wrap_by_words(text, num_words): # for the sanity check
            words = text.split(' ')
            lines = [' '.join(words[i:i + num_words]) for i in range(0, len(words), num_words)]
            return '\n'.join(lines)

        # Here we loop over random generation process to produce several captions
        list_ms_texts, list_combined_text, list_sanity_texts = [], [], []
        for caption_id in range(AUGMENTATION_PORTION):

            (binning_detial, generated_motion_description_non_agg, generated_motion_description,
             start_time, end_time) = generate_text_HumanML3D(id, motion_path, root_euler_path, motion_stats=motion_stats, ablations=[])

            generated_motion_description = generated_motion_description.strip()
            generated_motion_description_non_agg = generated_motion_description_non_agg.strip()
            len_array.append(len(nltk.word_tokenize(generated_motion_description)))

            if generated_motion_description.strip() == '':
                continue

            current_ms_text = f'{generated_motion_description}#None/None#{start_time:.1f}#{end_time:.1f}'
            list_ms_texts.append(current_ms_text)

            wrapped_motion_description = wrap_by_words((generated_motion_description), 15)  # Break lines at 10 words
            wrapped_motion_description_non_agg = wrap_by_words((generated_motion_description_non_agg), 15)

            current_sanity_text = (f'Start: {start_time:.1f}s ---- End: {end_time:.1f}s' + '\n' * 3 +  "Binning details:\n" +
                                   binning_detial + '\n' * 4 + 'NonAggregated Motioncodes:\n\n' +
                           (wrapped_motion_description_non_agg) + '\n' * 4 + 'Aggregated Motioncodes:\n\n' + (wrapped_motion_description))
            list_sanity_texts.append(current_sanity_text)

            counter_gen += 1


        # Save Sanity
        all_sanity_text = "\n\n".join(list_sanity_texts)
        f_text = open(save_path_text_sanity, 'w', encoding="utf-8")
        f_text.write(f'{all_sanity_text}\n\nHuman annotation:\n{org_caption}')
        f_text.flush()
        f_text.close()

        # Save MotionScript only captions
        all_ms_text = "\n".join(list_ms_texts)
        f_text = open(save_path_text_ms, 'w', encoding="utf-8")
        f_text.write(all_ms_text)
        f_text.flush()
        f_text.close()



        all_combined = "\n".join(list_ms_texts) + '\n' + org_caption
        all_combined = all_combined.strip()
        # if generated_motion_description != '':
        #     combined_text = f'{generated_motion_description}#None/None#{start_time:.1f}#{end_time:.1f}\n{org_caption}'
        # else:
        #     combined_text = org_caption
        f_text = open(save_path_text_mshml3d, 'w', encoding="utf-8")
        f_text.write(all_combined)
        f_text.flush()
        f_text.close()


    if False: # except:
        problem_list.append(id)
        print("PROBLEM LIST:", problem_list)
    break


# apply the satistic analysis step 2
if motion_stats:
    stat_save_dir = 'out_temp/statistics/'
    if not os.path.exists(stat_save_dir):
        os.mkdir(stat_save_dir)
    pickle.dump(all_motion_stats, open(os.path.join(stat_save_dir, 'all_motion_stats_step1.pkl'), 'wb'))
    captioning_py.motioncode_stat_analysis_step2_visualization(all_motion_stats, stat_save_dir)
    print(f"{'*'*5}Statistical Analysis{'*'*5}\nSaved in: {stat_save_dir}")

# Restore stdout to its original setting
sys.stdout = original_stdout



# print("PROBLEM LIST:\n", problem_list)
print(counter_gen, counter_line)
len_array = np.array(len_array)
mean = np.mean(len_array)
std_dev = np.std(len_array)
print('Mean:', mean)
print('Standard Deviation:', std_dev)
