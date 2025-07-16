

import os
from tqdm import tqdm
import json
import numpy as np
import pickle
from tabulate import tabulate

import text2pose.config as config
import text2pose.utils as utils
import csv
import tqdm

### SETUP
################################################################################

# 1. First, we need to load HumanML3D index.csv which provides the mapping between HML3D and AMASS
PATH2AMASS_DATASET = '../data/amass_data'
PATH2_HML3D_POSES = '../data/HML3D_Pose_data'

def HML3D_time_2_AMASS_time(path2_amas_used, hml3d_start_frame, hml3d_end_frame):
    HML3D_fps = 20
    loaded_amass_npz = np.load(path2_amas_used, allow_pickle=True)

    if 'humanact12' in path2_amas_used:
        amass_fps = HML3D_fps
        hml3d_end_frame = len(loaded_amass_npz)
    else:
        amass_fps = loaded_amass_npz['mocap_framerate']
    down_sample = int(amass_fps / HML3D_fps)

    # Based on HumanML3D preprocessing.
    if 'humanact12' not in path2_amas_used:
        if 'Eyes_Japan_Dataset' in path2_amas_used:
            hml3d_start_frame += 3 * HML3D_fps # data = data[3 * fps:]
            hml3d_end_frame   += 3 * HML3D_fps
        if 'MPI_HDM05' in path2_amas_used:
            hml3d_start_frame += 3 * HML3D_fps # data = data[3 * fps:]
            hml3d_end_frame   += 3 * HML3D_fps
        if 'TotalCapture' in path2_amas_used:
            hml3d_start_frame += 1 * HML3D_fps # data = data[1 * fps:]
            hml3d_end_frame   += 1 * HML3D_fps
        if 'MPI_Limits' in path2_amas_used:
            hml3d_start_frame += 1 * HML3D_fps # data = data[1 *
            hml3d_end_frame   += 1 * HML3D_fps
        if 'Transitions_mocap' in path2_amas_used:
            hml3d_start_frame += 0.5 * HML3D_fps # data = data[int(0.5 * fps):]
            hml3d_end_frame   += 0.5 * HML3D_fps
        # data = data[start_frame:end_frame]

    return {
        'AMASS_start_frame': hml3d_start_frame * down_sample,
        'AMASS_end_frame': hml3d_end_frame * down_sample,
        'AMASS_start_time': (hml3d_start_frame * down_sample) / amass_fps,
        'AMASS_end_time': (hml3d_end_frame * down_sample) / amass_fps
    }


def HML3D_AMASS():


    # 1. We first extract a mapping between HumanML3D and AMASS dataset

    # Header: source_path start_frame end_frame new_name
    file_path = 'index(HumanML3D).csv'
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        rows_list = []
        HML3D_IDs = dict()
        # AMASS_2_HML3D_ID = dict()


        for row in csv_reader:
            rows_list.append(row)
            feat_p = row['source_path']
            HML3D_id = row['new_name'][:-4] # remove '.npy'
            HML3D_IDs[HML3D_id] = {'feat_p': feat_p,
                                          'start_frame': row['start_frame'],
                                          'end_frame': row['end_frame']}
            # AMASS_2_HML3D_ID[feat_p] = HML3D_id

    # 2. BABEL is defined relative to the AMASS

    HMLD3D_2_BABEL = dict()
    counter = 0
    for hml3d_id in tqdm.tqdm(HML3D_IDs):
        # if counter>400:
        #     break
        # counter+=1
        AMASS_rel_hml3d_feat_p = HML3D_IDs[hml3d_id]['feat_p']
        hml3d_start_frame  = int(HML3D_IDs[hml3d_id]['start_frame'])
        hml3d_end_frame    = int(HML3D_IDs[hml3d_id]['end_frame'])
        # path2_amas_used = hml3d_feat_p

        if 'humanact12' in AMASS_rel_hml3d_feat_p:
            HMLD3D_2_BABEL[hml3d_id] = {'sequence_labels': None, 'frame_labels': None}
            continue

        fixed_AMASS_rel_HML3D_feat_p = PATH2AMASS_DATASET + AMASS_rel_hml3d_feat_p[len('./pose_data'):-len('.npy')] + '.npz'
        if 'humanact12' in AMASS_rel_hml3d_feat_p:
            fixed_AMASS_rel_HML3D_feat_p = PATH2_HML3D_POSES +  AMASS_rel_hml3d_feat_p[len('./pose_data'):]
        AMASS_time_info = HML3D_time_2_AMASS_time(path2_amas_used=fixed_AMASS_rel_HML3D_feat_p,
                                            hml3d_start_frame=hml3d_start_frame,
                                            hml3d_end_frame=hml3d_end_frame)

        # get path correspondance in BABEL
        fixed_BABEL_rel_AMASS_path =  AMASS_rel_hml3d_feat_p[len('./pose_data/'):-len('.npy')] + '.npz'
        dname = AMASS_rel_hml3d_feat_p[len('./pose_data/'):].split('/')[0]
        bname = amass_to_babel_subdir[dname]

        if bname == '':
            babel_labels = {'sequence_labels': None, 'frame_labels': None} #  '__' + dname + '__'
        else:
            fixed_BABEL_rel_AMASS_path = '/'.join([bname] + (AMASS_rel_hml3d_feat_p[len('./pose_data/'):-len('.npy')] + '.npz').split('/')[1:])
            babel_labels = get_babel_label_motionscript(amass_rel_path=fixed_BABEL_rel_AMASS_path,
                                                        time_info=AMASS_time_info)
        HMLD3D_2_BABEL[hml3d_id] = babel_labels

    return HMLD3D_2_BABEL




# load BABEL
l_babel_dense_files = ['train', 'val', 'test']
l_babel_extra_files = ['extra_train', 'extra_val']

babel = {}
for file in l_babel_dense_files:
    babel[file] = json.load(open(os.path.join(config.BABEL_LOCATION, file+'.json')))
    
for file in l_babel_extra_files:
    babel[file] = json.load(open(os.path.join(config.BABEL_LOCATION, file+'.json')))    

# load PoseScript
dataID_2_pose_info = utils.read_posescript_json("ids_2_dataset_sequence_and_frame_index.json")

# AMASS/BABEL path adaptation
amass_to_babel_subdir = {
    'ACCAD': 'ACCAD/ACCAD',
    'BMLhandball': '', # not available
    'BMLmovi': 'BMLmovi/BMLmovi',
    'BioMotionLab_NTroje': 'BMLrub/BioMotionLab_NTroje',
    'CMU': 'CMU/CMU',
    'DFaust_67': 'DFaust67/DFaust_67',
    'DanceDB': '', # not available
    'EKUT': 'EKUT/EKUT',
    'Eyes_Japan_Dataset': 'EyesJapanDataset/Eyes_Japan_Dataset',
    'HumanEva': 'HumanEva/HumanEva',
    'KIT': 'KIT/KIT',
    'MPI_HDM05': 'MPIHDM05/MPI_HDM05',
    'MPI_Limits': 'MPILimits/MPI_Limits',
    'MPI_mosh': 'MPImosh/MPI_mosh',
    'SFU': 'SFU/SFU',
    'SSM_synced': 'SSMsynced/SSM_synced',
    'TCD_handMocap': 'TCDhandMocap/TCD_handMocap',
    'TotalCapture': 'TotalCapture/TotalCapture',
    'Transitions_mocap': 'Transitionsmocap/Transitions_mocap',
}


### GET LABELS
################################################################################

def get_babel_label(amass_rel_path, frame_id):

    # get path correspondance in BABEL
    dname = amass_rel_path.split('/')[0]
    bname = amass_to_babel_subdir[dname]
    if bname == '': 
        return '__'+dname+'__'
    babel_rel_path = '/'.join([bname]+amass_rel_path.split('/')[1:])

    # look for babel annotations
    babelfs = []
    for f in babel.keys():
        for s in babel[f].keys():
            if babel[f][s]['feat_p'] == babel_rel_path:
                babelfs.append((f,s))

    if len(babelfs) == 0:
        return None

    # convert frame id to second
    seqdata = np.load(os.path.join(config.AMASS_FILE_LOCATION, amass_rel_path))
    framerate = seqdata['mocap_framerate']
    t = frame_id / framerate

    # read babel annotations
    labels = []
    for f,s in babelfs:
        if not 'frame_ann' in  babel[f][s]:
            continue
        if babel[f][s]['frame_ann'] is None: 
            continue
        babel_annots = babel[f][s]['frame_ann']['labels']
        for i in range(len(babel_annots)):
            if t >= babel_annots[i]['start_t'] and t <= babel_annots[i]['end_t']:
                labels.append( (babel_annots[i]['raw_label'], babel_annots[i]['proc_label'], babel_annots[i]['act_cat']) )

    return labels

def get_babel_label_motionscript(amass_rel_path, time_info):

    # # get path correspondance in BABEL
    # dname = amass_rel_path.split('/')[0]
    # bname = amass_to_babel_subdir[dname]
    # if bname == '':
    #     return '__'+dname+'__'
    # babel_rel_path = '/'.join([bname]+amass_rel_path.split('/')[1:])
    babel_rel_path = amass_rel_path # check if we need to do any modification

    if 'humanact12' in babel_rel_path:
        return {'sequence_labels': None, 'frame_labels': None} # since BABEL doesn't support Humanact12

    # look for babel annotations
    babelfs = []
    for f in babel.keys():
        for s in babel[f].keys():
            if babel[f][s]['feat_p'] == babel_rel_path:
                babelfs.append((f,s))

    if len(babelfs) == 0:
        return {'sequence_labels': None, 'frame_labels': None}
    # convert frame id to second
    # seqdata = np.load(os.path.join(config.AMASS_FILE_LOCATION, amass_rel_path))
    # framerate = seqdata['mocap_framerate']
    # t = frame_id / framerate

    # read babel annotations
    frame_labels = []
    sequence_labels = []
    for f,s in babelfs:
        # inconsistency between train/val and extea_train_val dic
        _is_extra = 'extra' in f
        frame_ann_dic_key = 'frame_anns' if _is_extra else 'frame_ann'
        if frame_ann_dic_key in  babel[f][s] :
            if babel[f][s][frame_ann_dic_key] is not None:
                if _is_extra:
                    babel_annots = [label for sublist in babel[f][s][frame_ann_dic_key] for label in sublist['labels']]
                else:
                    babel_annots = babel[f][s][frame_ann_dic_key]['labels']
                for i in range(len(babel_annots)):
                    overlap = max(0, min(babel_annots[i]['end_t'], time_info['AMASS_end_time']) - max(babel_annots[i]['start_t'], time_info['AMASS_start_time']))
                    percentage_covered = (overlap / (time_info['AMASS_end_time'] - time_info['AMASS_start_time'])) * 100 if overlap > 0 else 0
                    if overlap>0:
                        frame_labels.append( {'raw_label': babel_annots[i]['raw_label'],
                                              'proc_label': babel_annots[i]['proc_label'],
                                              'act_cat': babel_annots[i]['act_cat'],
                                             'HML3D_start_t':  babel_annots[i]['start_t'] - time_info['AMASS_start_time'], # time w.r.t. the HML3D sample time
                                              'HML3D_end_t': babel_annots[i]['end_t'] - time_info['AMASS_start_time']
                                              })

        seq_ann_dic_key = 'seq_anns' if _is_extra else 'seq_ann'
        if seq_ann_dic_key in babel[f][s]:
            if babel[f][s][seq_ann_dic_key] is not None:
                if _is_extra:
                    babel_annots = [label for sublist in babel[f][s][seq_ann_dic_key] for label in sublist['labels']]
                else:
                    babel_annots = babel[f][s][seq_ann_dic_key]['labels']
            for i in range(len(babel_annots)):
                # overlap = max(0, min(babel_annots[i]['end_t'], time_info['AMASS_end_time']) - max(babel_annots[i]['start_t'], time_info['AMASS_start_time']))
                # percentage_covered = (overlap / (time_info['AMASS_end_time'] - time_info['AMASS_start_time'])) * 100 if overlap > 0 else 0
                sequence_labels.append( {'raw_label': babel_annots[i]['raw_label'],
                                      'proc_label': babel_annots[i]['proc_label'],
                                      'act_cat': babel_annots[i]['act_cat'],
                                     'HML3D_start_t': 0, # time w.r.t. the HML3D sample time, whole sequence
                                      'HML3D_end_t': time_info['AMASS_end_time'] - time_info['AMASS_start_time']
                                      })

    return {'sequence_labels': sequence_labels,
            'frame_labels': frame_labels}


# gather labels for all motions in HumanML3D that come from AMASS and have BABEL labels
HML3D_BABELed = HML3D_AMASS()
save_filepath = os.path.join(config.POSESCRIPT_LOCATION, "babel_labels_for_motionscript.pkl")
with open(save_filepath, 'wb') as f:
    pickle.dump(HML3D_BABELed, f)
















'''
# gather labels for all poses in PoseScript that come from AMASS
counter = 0
babel_labels_for_posescript = {}
for dataID in tqdm.tqdm(dataID_2_pose_info):
    pose_info = dataID_2_pose_info[dataID]
    if pose_info[0] == "AMASS":
        babel_labels_for_posescript[dataID] = get_babel_label(pose_info[1], pose_info[2])
    if counter>400:
        break
    counter +=1

# display some stats
table = []
table.append(['None', sum([v is None for v in babel_labels_for_posescript.values()])])
table.append(['BMLhandball', sum([v=='__BMLhandball__' for v in babel_labels_for_posescript.values()])])
table.append(['DanceDB', sum([v=='__DanceDB__' for v in babel_labels_for_posescript.values()])])
table.append(['0 label', sum([ (isinstance(v,list) and len(v)==0) for v in babel_labels_for_posescript.values()])])
table.append(['None label', sum([ (isinstance(v,list) and len(v)>=1 and v[0][0] is None) for v in babel_labels_for_posescript.values()])])
table.append(['1 label', sum([ (isinstance(v,list) and len(v)==1 and v[0][0] is not None) for v in babel_labels_for_posescript.values()])])
table.append(['>1 label',sum([ (isinstance(v,list) and len(v)>=2 and v[0][0] is not None) for v in babel_labels_for_posescript.values()])])
print(tabulate(table, headers=["Label", "Number of poses"]))

# save
save_filepath = os.path.join(config.POSESCRIPT_LOCATION, "babel_labels_for_posescript.pkl")
with open(save_filepath, 'wb') as f:
    pickle.dump(babel_labels_for_posescript, f)
print("Saved", save_filepath)
'''