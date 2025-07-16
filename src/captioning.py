
import math
# requires at least Python 3.6 (order preserved in dicts)

import os, sys, time
import pickle, json
import random
import copy
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpmath.libmp.libmpf import special_str
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from tabulate import tabulate

from text2pose.posescript.posecodes import POSECODE_OPERATORS, MOTIONCODE_OPERATORS, TIMECODE_OPERATORS, distance_between_joint_pairs
from text2pose.posescript.captioning_data import *
from text2pose.posescript.captioning_data_ablation import *
from MS_Algorithms import min_samples_to_cover

from text2pose.posescript.MS_Algorithms import single_path_finder
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################################
## UTILS
################################################################################

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def list_remove_duplicate_preserve_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# (SMPL-H) skeleton (22 main body + 2*15 hands), from https://meshcapade.wiki/SMPL#skeleton-layout
ALL_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',

    #     Added:
    'orientation',
    'translation'
]

# Joints that are actually useful for the captioning pipeline
VIRTUAL_JOINTS = ["left_hand", "right_hand", "torso"] # ADD_VIRTUAL_JOINT
# JOINT_NAMES = ALL_JOINT_NAMES[:22] +  + ['left_middle2', 'right_middle2'] + VIRTUAL_JOINTS
JOINT_NAMES = ALL_JOINT_NAMES[:22] + ALL_JOINT_NAMES[-2:] + ['left_middle2', 'right_middle2'] + VIRTUAL_JOINTS


JOINT_NAMES2ID = {jn:i for i, jn in enumerate(JOINT_NAMES)}

# </Payam>
swapped_JOINT_NAMES2ID = {value: key for key, value in JOINT_NAMES2ID.items()}

# Interpretation set (interpretations from the posecode operators + (new
# distinct) interpretations from the set of super-posecodes)
# (preserving the order of the posecode operators interpretations, to easily
# convert operator-specific interpretation ids to global interpretation ids,
# using offsets ; as well as the order of super-posecode interpretations, for 
# compatibility accross runs)
INTERPRETATION_SET = flatten_list([p["category_names"] for p in POSECODE_OPERATORS_VALUES.values()])
sp_interpretation_set = [v[1][1] for v in SUPER_POSECODES if v[1][1] not in INTERPRETATION_SET]
INTERPRETATION_SET += list_remove_duplicate_preserve_order(sp_interpretation_set)
INTPTT_NAME2ID = {intptt_name:i for i, intptt_name in enumerate(INTERPRETATION_SET)}

# Data to reverse subjects & select template sentences
OPPOSITE_CORRESP_ID = {INTPTT_NAME2ID[k]:INTPTT_NAME2ID[v] for k, v in OPPOSITE_CORRESP.items()}
OK_FOR_1CMPNT_OR_2CMPNTS_IDS = [INTPTT_NAME2ID[n] for n in OK_FOR_1CMPNT_OR_2CMPNTS]



################################################################################
## Motioncodes

# Spatials
INTERPRETATION_SET_MOTION = flatten_list([p["category_names"] for p in MOTIONCODE_OPERATORS_VALUES.values()])

# Temporals: Since we assume all velocty interpretations are the same, however, they might have
# different thresholds for different motion codes
GLOBAL_VELOCITY_INTTPT_OFFSET = len(INTERPRETATION_SET_MOTION)
INTERPRETATION_SET_MOTION += flatten_list ([ next(iter(MOTIONCODE_OPERATORS_VALUES.values()))["category_names_velocity"] ] )

# sp_interpretation_set_spatial = [v[1][1] for v in SUPER_MOTIONCODE if v[1][1] not in INTERPRETATION_SET]      todo: fix this for super-motion code
# INTERPRETATION_SET_SPATIAL += list_remove_duplicate_preserve_order(sp_interpretation_set_spatial)
INTPTT_NAME2ID_MOTION = {intptt_name:i for i, intptt_name in enumerate(INTERPRETATION_SET_MOTION)}

# Data to reverse subjects & select template sentences
OPPOSITE_CORRESP_ID_MOTIONCODES = {INTPTT_NAME2ID_MOTION[k]:INTPTT_NAME2ID_MOTION[v] for k, v in OPPOSITE_CORRESP_MOTIONCODES.items()}
OK_FOR_1CMPNT_OR_2CMPNTS_IDS_MOTIONCODES = [INTPTT_NAME2ID_MOTION[n] for n in OK_FOR_1CMPNT_OR_2CMPNTS_MOTONCODES]




################################################################################
## Timecodes
INTERPRETATION_SET_TIME = flatten_list([p["category_names"] for p in TIMECODE_OPERTATOR_VALUES.values()])
INTPTT_NAME2ID_TIME = {intptt_name:i for i, intptt_name in enumerate(INTERPRETATION_SET_TIME)}



################################################################################
def Motion_Path_Finder(p_interpretations, p_queries):

    '''
    def single_path_finder(signal):
        current_val = 0
        start_index = 0
        end_index = 0
        max_prev = {'start': 0, 'end': 0, 'transitions': 0, 'speed': 0}

        result_list = []

        while( end_index<len(signal)):

            if current_val == 0:
                if signal[end_index] == 0:
                    start_index = end_index

                elif signal[end_index] > 0: #signal[end_index-1]: # Increasing
                    # State Change
                    # Update value
                    current_val += 1
                    # Store max_prev
                    # Here we don't
                    # result_list.append(max_prev)
                    # Jump Stationary moments
                    start_index = end_index
                    transitions = current_val
                    New_Speed = current_val / (end_index - start_index + 1)
                    max_prev = {'start': start_index, 'end': end_index, 'transitions': transitions, 'speed': New_Speed}


                elif signal[end_index] < 0: # signal[end_index-1]: # Decreasing order
                    # State Change
                    # Update value
                    current_val += -1
                    # Skip stationary movements
                    start_index = end_index

                    transitions = current_val
                    New_Speed = current_val / (end_index - start_index + 1)
                    max_prev = {'start': start_index, 'end': end_index, 'transitions': transitions, 'speed': New_Speed}


            elif current_val > 0: # Increasing order

                if signal[end_index] == 0:
                    pass

                elif signal[end_index] > 0: # signal[end_index - 1]:  # Increasing
                    # No Change
                    # Update value
                    current_val += 1
                    # New_Worth = current_val / (end_index - start_index + 1)
                    # if abs(max_prev['worth']) <= abs(New_Worth):
                    #     max_prev = {'satrt': start_index, 'end': end_index, 'worth': New_Worth}
                    Current_transitions = current_val
                    Current_Speed = current_val / (end_index - start_index + 1)
                    if abs(max_prev['transitions']) < abs(Current_transitions):
                        max_prev = {'start': start_index, 'end': end_index, 'transitions': Current_transitions, 'speed': Current_Speed}



                elif signal[end_index] < 0: # signal[end_index - 1]:  # Decreasing order
                    # State Change
                    # Update value from start_index ---> end_index-1
                    end_before_current_move = end_index-1
                    while start_index < end_index-1: # up to start the new direction
                        if signal[start_index]== +1:
                            current_val -= +1

                        # New_Worth = current_val / ( (end_index-1) - start_index )
                        #
                        # if abs(max_prev['worth']) <= abs(New_Worth):
                        #     max_prev = {'satrt': start_index + 1, 'end': end_index-1, 'worth': New_Worth}

                        Current_transitions = current_val
                        Current_Speed = current_val / ( (end_index-1) - start_index )
                        if (abs(max_prev['transitions']) < abs(Current_transitions)) or \
                            ( abs(max_prev['transitions']) == abs(Current_transitions) and abs(max_prev['speed']) < abs(Current_Speed)) :
                            max_prev = {'start': start_index + 1, 'end': end_index-1, 'transitions': Current_transitions,
                                        'speed': Current_Speed}


                        start_index += 1

                    result_list.append(max_prev)
                    start_index = end_index
                    max_prev =  {'start': start_index, 'end': end_index, 'transitions': -1, 'speed': -1} # opposite val
                    current_val = -1

            # Decreasing flow
            elif current_val < 0: #todo: fix the following

                if signal[end_index] == 0:
                    pass

                elif signal[end_index] > 0: # signal[end_index - 1]:  # Increasing
                    # State Change
                    # Update value

                    # Todo: here we should loop start_index to end_index to consider the other side of the path

                    while start_index < end_index-1:
                        if signal[start_index] < 0: # else = signal_start_index] = 0
                            current_val -= -1

                        # Update max prev
                        # New_Worth = current_val / ( (end_index-1) - start_index) #Exclusive start
                        # if abs(max_prev['worth']) <= abs(New_Worth): # for 1/1, 2/2, ...  We should also consider cval comparison
                        #     max_prev = {'satrt': start_index + 1, 'end': end_index-1, 'worth': New_Worth}
                        Current_transitions = current_val
                        Current_Speed = current_val / ((end_index - 1) - start_index)
                        if (abs(max_prev['transitions']) < abs(Current_transitions)) or \
                                (abs(max_prev['transitions']) == abs(Current_transitions) and
                                 abs(max_prev['speed']) < abs(Current_Speed)):
                            max_prev = {'start': start_index + 1, 'end': end_index-1, 'transitions': Current_transitions,
                                        'speed': Current_Speed}



                        start_index += 1

                    result_list.append(max_prev)
                    start_index = end_index
                    max_prev = {'start': start_index, 'end': end_index, 'transitions': +1, 'speed': 1} # opposite val
                    current_val = +1

                elif signal[end_index] < 0: # signal[end_index - 1]:  # Decreasing
                    # No Change
                    # Update value
                    current_val += -1
                    # New_Worth = current_val / (end_index - start_index + 1)
                    # if abs(New_Worth) >= abs(max_prev['worth']):
                    #     max_prev = {'satrt': start_index, 'end': end_index, 'worth': New_Worth}

                    Current_transitions = current_val
                    Current_Speed = current_val / (end_index - start_index + 1)
                    if abs(max_prev['transitions']) < abs(Current_transitions): # it would be always the case unless we ad more conditions
                        max_prev = {'start': start_index, 'end': end_index, 'transitions': Current_transitions,
                                    'speed': Current_Speed}

            # ---------------STEP---------------
            end_index += 1
        # Tod: ending

        if start_index<end_index:
            if current_val > 0:
                while start_index < end_index - 1:  # up to start the new direction
                    if signal[start_index] == +1:
                        current_val -= +1
                    Current_transitions = current_val
                    Current_Speed = current_val / ((end_index - 1) - start_index)
                    if (abs(max_prev['transitions']) < abs(Current_transitions)) or \
                            (abs(max_prev['transitions']) == abs(Current_transitions) and abs(max_prev['speed']) < abs(
                                Current_Speed)):
                        max_prev = {'start': start_index + 1, 'end': end_index - 1, 'transitions': Current_transitions,
                                    'speed': Current_Speed}
                    start_index += 1

            if current_val < 0:
                while start_index < end_index - 1:
                    if signal[start_index] < 0:  # else = signal_start_index] = 0
                        current_val -= -1
                    Current_transitions = current_val
                    Current_Speed = current_val / ((end_index - 1) - start_index)
                    if (abs(max_prev['transitions']) < abs(Current_transitions)) or \
                            (abs(max_prev['transitions']) == abs(Current_transitions) and
                             abs(max_prev['speed']) < abs(Current_Speed)):
                        max_prev = {'start': start_index + 1, 'end': end_index - 1, 'transitions': Current_transitions,
                                    'speed': Current_Speed}

                    start_index += 1

            result_list.append(max_prev)






        # print('Final Result \n')
        # print('*' * 20)
        # print(f'Signal: \n{signal}\n')
        # print(f'ResultS:\n {result_list}')
        # print('*'*20)
        return  result_list
   '''

    from MS_Algorithms import single_path_finder

    result_matrix = dict()
    js_motion_detected = dict()
    for p_kind in p_interpretations:

        js_motion_detected[p_kind] = []
        shape = p_interpretations[p_kind].shape  # [nb_poses, nb_joints]
        for js in range(shape[1]): # for over joints




            time_series_signal = p_interpretations[p_kind][:, js].cpu().numpy()

            delta_signal = [time_series_signal[i + 1] - time_series_signal[i] for i in range(len(time_series_signal) - 1)]
            delta_signal = [0] + delta_signal # to initiate with 0

            # delta_signal = [0, 0 , 0, 1, 1, 1, 0, -1, 1, 0 -1, 0, -1, -1, +1, 0, +1, 0, +1, +1, 0, 0, 0]

            list_of_motion_time_window = single_path_finder(delta_signal,time_series_signal )
            js_motion_detected[p_kind] += [{'js': js, 'motions': list_of_motion_time_window}]
    print()
#     Todo: Now we have motions and we need to convert them back to their name from index and visualize them and get ready for superposecodes

    def motion_code2text(motion_codes):
        str_out = ""
        for p_kind in motion_codes:
            for js in range(len(motion_codes[p_kind])):
                jids = p_queries[p_kind]['joint_ids'][js]
                focus_body_part = p_queries[p_kind]['focus_body_part'][js]
                jts_names = [swapped_JOINT_NAMES2ID[key] for key in jids.numpy()]
                class_change = motion_codes[p_kind][js]['motions']
                str_out += '*'*80 + '\n'
                str_out += (f'**Class: {p_kind} --- Body Part: {focus_body_part} ---> Joints({jts_names}\n')
                for change in class_change:
                    if change['intensity']==0:
                        continue
                    str_out += (f"start: {change['start']}, end: {change['end']}, intensity: {change['intensity']}, velocity: {change['velocity']:.2f}\n")
                str_out += ("-"*80 )
                str_out += ('\n\n')
        file = open('out_temp/current_motion_codes.txt', 'w')
        file.write(str_out)
        file.close()
    motion_code2text(js_motion_detected)
    print()
    p_queries
    return js_motion_detected



################################################################################
## MAIN
################################################################################





def main(coords, save_dir, babel_info=False, simplified_captions=False,
        apply_transrel_ripple_effect=True, apply_stat_ripple_effect=True,
        random_skip=True, motion_tracking=False, verbose=True, ablations=[]):


    # we apply bining step here to specify the chronological order
    # Todo: fix this and move it outside of this function to be usable by others.
    total_frames, fps = coords.shape[0], 20
    thresholds = (TIMECODE_OPERTATOR_VALUES['ChronologicalOrder']['category_thresholds'])
    bin_size = int( min((thresholds[i + 1] - thresholds[i]) for i in range(len(thresholds) - 1)) * fps )
    nb_binds = int (total_frames // bin_size) + 1
    max_range_bins = (int( (TIMECODE_OPERTATOR_VALUES['ChronologicalOrder']['category_thresholds'][-1] * fps)) // bin_size) + 1
    Time_Bin_Info = {'bin_size': bin_size,
                     'max_range_bins': max_range_bins,
                     'nb_binds': nb_binds,
                     'total_frames': total_frames}

    if "intensity" in ablations:
        # ENHANCE_TEXT_1CMPNT_Motion_set(ENHANCE_TEXT_1CMPNT_Motion__ABLATION_INTENSITY)
        # ENHANCE_TEXT_2CMPNT_Motion_set(ENHANCE_TEXT_2CMPNT_Motion__ABLATION_INTENSITY)
        # ENHANCE_TEXT_1CMPNT_Motion
        # ENHANCE_TEXT_2CMPNT_Motion
        pass
    if "velocity" in ablations:
        # VELOCITY_ADJECTIVES = VELOCITY_ADJECTIVES_ABLATION
        pass
    if "chronological" in ablations:
        # CHRONOLOGICAL_ORDER_ADJECTIV = CHRONOLOGICAL_ORDER_ADJECTIVE__ABLATION
        pass
    # This doesn't work and I need to do it manually.

    if verbose: print("Formating input...")
    # Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
    coords = prepare_input(coords)
    # Prepare posecode queries
    # (hold all info about posecodes, essentially using ids)

    # Posecodes:
    p_queries = prepare_posecode_queries()
    sp_queries = prepare_super_posecode_queries(p_queries)

    # Motioncodes
    m_queries = prepare_motioncode_queries()

    if verbose: print("Eval & interprete & elect eligible posecodes...")
    # Eval & interprete & elect eligible elementary posecodes
    p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries,
                                                       request='Low_Level_intptt' if motion_tracking else 'Default',
                                                       verbose=verbose)

    # list_of_Returned_Results = Motion_Path_Finder(p_interpretations, p_queries)
    m_interpretations, m_eligibility = infer_motioncodes(coords, p_interpretations, p_queries, sp_queries, m_queries,
                                                         request='Low_Level_intptt' if motion_tracking else 'Default',
                                                         verbose=verbose)



    #Todo: save the m_interpretations for style transfer project.
    # save and return
    # made a copy of this





    # list_of_Returned_Results = Motion_Path_Finder(p_interpretations, p_queries)

    #Todo: --------------------- strategy
    # Here we provide a snippet to adjust eligbilty w.r.t. several strategy.
    # 1. It may consider excluding some of the rules to study the contribution
    #       of different sets of rules
    # 2. Longest sequence that contains a limited number of motioncodes e.g 5.
    # 3. It may consider to take a heuristic to pick longer sequences but only
    #       a limited set of joints e.g. left elbow, elbows, etc.

    one = True          # Eligibility Adjustment
    two = True         #
    three = False       # Export all the motioncodes to a json file
    # 1. Eligibility Adjustment
    if one:
        set_of_accepted_mkinds = [  'angular',
                                    'proximity',
                                    # 'spatial_relation_x', # no Spatials
                                    # 'spatial_relation_y',
                                    # 'spatial_relation_z',
                                    'displacement_x',
                                    'displacement_y',
                                    'displacement_z',
                                    'rotation_pitch',
                                    'rotation_roll',
                                    'rotation_yaw',
                                 ]
        for m_kind in m_eligibility:
            if m_kind not in set_of_accepted_mkinds:
        #         Turn all the detected motioncodes to uneligible, both spatial and temporal.
                    for js_id in range(len(m_eligibility[m_kind])):
                        for m_index in range(len(m_eligibility[m_kind][js_id])):
                            m_eligibility[m_kind][js_id][m_index] = (torch.tensor(0), torch.tensor(0))

    if two:
        m_kind = 'proximity'
        for js_id in range(len(m_eligibility[m_kind])):
            for m_index in range(len(m_eligibility[m_kind][js_id])):
                m_eligibility[m_kind][js_id][m_index] = (torch.tensor(1), torch.tensor(1))



    # save
    # saved_filepath = os.path.join(save_dir, "posecodes_intptt_eligibility.pt")
    # torch.save([p_interpretations, p_eligibility, INTPTT_NAME2ID], saved_filepath)
    # print("Saved file:", saved_filepath)

    # Format posecode for future steps & apply random skip
    # At this stage, we also have Support type III when a posecode is defined for
    # being used by motioncodes. Now we should make them uneligibile.
    # Todo: we may reimplement it later by defining rules, I guess I did it. Double-check the query on captioning_data file.
    # I think I already did it by support type II. Only the root translation ( ?: and orientations) are none support posecodes.
    # list_of_support_III = ['position_x', 'position_y', 'position_z', 'orientation_pitch', 'orientation_roll', 'orientation_yaw']
    # for pose_kind in list_of_support_III:
    #     p_eligibility[pose_kind] = torch.zeros_like(p_eligibility[pose_kind])

    if verbose: print("Formating posecodes...")
    posecodes, posecodes_skipped = format_and_skip_posecodes(p_interpretations,
                                                            p_eligibility,
                                                            p_queries,
                                                            sp_queries,
                                                            random_skip,
                                                            verbose = verbose)

    motioncodes, motioncodes_skipped = format_and_skip_motioncodes(m_interpretations,
                                                             m_eligibility,
                                                             m_queries,
                                                             None,
                                                             random_skip,
                                                             verbose=verbose)

    binning_details, motioncodes4vis = motioncodes_sanity_check(motioncodes, Time_Bin_Info)


    # save
    saved_filepath = os.path.join(save_dir, "posecodes_formated.pt")
    # torch.save([posecodes, posecodes_skipped], saved_filepath)
    print("Saved file:", saved_filepath)

    # Aggregate & discard posecodes (leverage relations)
    if verbose: print("Aggregating posecodes...")
    posecodes = aggregate_posecodes(posecodes,
                                    simplified_captions,
                                    apply_transrel_ripple_effect,
                                    apply_stat_ripple_effect)
    # motioncodes = pickle.load(open('debug-motioncode2.pk', 'rb'))

    # Define non aggregated motion codes for the visualization

    non_agg_motioncodes = motioncodes_agg = aggregate_motioncodes({'p_interpretations': p_interpretations,
                                                           'p_queries': p_queries}, # for detecting active joints
                                            motioncodes,
                                            Time_Bin_Info,
                                            simplified_captions,
                                            apply_transrel_ripple_effect,
                                            apply_stat_ripple_effect,
                                            agg_deactivated=True)

    motioncodes_agg = aggregate_motioncodes({'p_interpretations': p_interpretations,
                                                           'p_queries': p_queries}, # for detecting active joints
                                            motioncodes,
                                            Time_Bin_Info,
                                            simplified_captions,
                                            apply_transrel_ripple_effect,
                                            apply_stat_ripple_effect,
                                            agg_deactivated=False)
    print("")


    # Here we should call a function to also
    # 1. Evaluate time connections inside motioncodes
    # 2. Interpret timecodes to classify time connections
    # 3. Skip and Format based on the eligibilities
    # 4. No Aggregation step is required.
    motioncodes_agg_t = infer_timecodeds(motioncodes_agg)
    non_agg_motioncodes_t = infer_timecodeds(non_agg_motioncodes)
    # save
    saved_filepath = os.path.join(save_dir, "posecodes_aggregated.pt")
    # torch.save(posecodes, saved_filepath)
    print("Saved file:", saved_filepath)

    # Produce descriptions
    if verbose: print("Producing descriptions...")
    # descriptions, determiners = convert_posecodes(posecodes, simplified_captions)

    motion_descriptions, motion_determiners = convert_motioncodes(posecodes,
                                                                  motioncodes_agg_t,
                                                                  Time_Bin_Info,
                                                                  simplified_captions)
    motion_descriptions_non_agg, motion_determiners_non_agg = convert_motioncodes(posecodes,
                                                                  non_agg_motioncodes_t,
                                                                  Time_Bin_Info,
                                                                  simplified_captions)
    print(" ".join(motion_descriptions))


    if babel_info:
        added_babel_sent = 0
        for i in range(len(descriptions)):
            if babel_info[i]:
                added_babel_sent += 1
                # consistence with the chosen determiner
                if babel_info[i].split()[0] == "They":
                    if determiners[i] == "his": babel_info[i] = babel_info[i].replace("They", "He").replace(" are ", " is ")
                    elif determiners[i] == "her": babel_info[i] = babel_info[i].replace("They", "She").replace(" are ", " is ")
                elif babel_info[i].split()[1] == "human":
                    if determiners[i] == "his": babel_info[i] = babel_info[i].replace(" human ", " man ")
                    elif determiners[i] == "her": babel_info[i] = babel_info[i].replace(" human ", " woman ")
            # eventually add the BABEL tag information
            descriptions[i] = babel_info[i] + descriptions[i]
        if verbose: print(f"Added {added_babel_sent} new sentences using information extracted from BABEL.")

    # # save
    # saved_filepath = os.path.join(save_dir, "descriptions.json")
    # descriptions = {i:descriptions[i] for i in range(len(descriptions))}
    # with open(saved_filepath, "w") as f:
    #     json.dump(descriptions, f, indent=4, sort_keys=True)
    # print("Saved file:", saved_filepath)
    # # torch.save(descriptions, saved_filepath)

    return (binning_details, motioncodes4vis, motion_descriptions_non_agg , motion_descriptions)


def main4scanline_motioncodes(coords, save_dir, babel_info=False, simplified_captions=False,
        apply_transrel_ripple_effect=True, apply_stat_ripple_effect=True,
        random_skip=True, motion_tracking=False, verbose=True, ablations=[]):


    # we apply bining step here to specify the chronological order
    # Todo: fix this and move it outside of this function to be usable by others.
    total_frames, fps = coords.shape[0], 20
    thresholds = (TIMECODE_OPERTATOR_VALUES['ChronologicalOrder']['category_thresholds'])
    bin_size = int( min((thresholds[i + 1] - thresholds[i]) for i in range(len(thresholds) - 1)) * fps )
    nb_binds = int (total_frames // bin_size) + 1
    max_range_bins = (int( (TIMECODE_OPERTATOR_VALUES['ChronologicalOrder']['category_thresholds'][-1] * fps)) // bin_size) + 1
    Time_Bin_Info = {'bin_size': bin_size,
                     'max_range_bins': max_range_bins,
                     'nb_binds': nb_binds,
                     'total_frames': total_frames}

    if "intensity" in ablations:
        # ENHANCE_TEXT_1CMPNT_Motion_set(ENHANCE_TEXT_1CMPNT_Motion__ABLATION_INTENSITY)
        # ENHANCE_TEXT_2CMPNT_Motion_set(ENHANCE_TEXT_2CMPNT_Motion__ABLATION_INTENSITY)
        # ENHANCE_TEXT_1CMPNT_Motion
        # ENHANCE_TEXT_2CMPNT_Motion
        pass
    if "velocity" in ablations:
        # VELOCITY_ADJECTIVES = VELOCITY_ADJECTIVES_ABLATION
        pass
    if "chronological" in ablations:
        # CHRONOLOGICAL_ORDER_ADJECTIV = CHRONOLOGICAL_ORDER_ADJECTIVE__ABLATION
        pass
    # This doesn't work and I need to do it manually.

    if verbose: print("Formating input...")
    # Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
    coords = prepare_input(coords)
    # Prepare posecode queries
    # (hold all info about posecodes, essentially using ids)

    # Posecodes:
    p_queries = prepare_posecode_queries()
    sp_queries = prepare_super_posecode_queries(p_queries)

    # Motioncodes
    m_queries = prepare_motioncode_queries()

    if verbose: print("Eval & interprete & elect eligible posecodes...")
    # Eval & interprete & elect eligible elementary posecodes
    p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries,
                                                       request='Low_Level_intptt' if motion_tracking else 'Default',
                                                       verbose=verbose)

    # list_of_Returned_Results = Motion_Path_Finder(p_interpretations, p_queries)
    m_interpretations, m_eligibility = infer_motioncodes(coords, p_interpretations, p_queries, sp_queries, m_queries,
                                                         request='Low_Level_intptt' if motion_tracking else 'Default',
                                                         verbose=verbose)

    motioncodes, motioncodes_skipped = format_and_skip_motioncodes(m_interpretations,
                                                                   m_eligibility,
                                                                   m_queries,
                                                                   None,
                                                                   random_skip,
                                                                   verbose=verbose)

    binning_details, motioncodes4vis = motioncodes_sanity_check(motioncodes, Time_Bin_Info)




    #Todo: save the m_interpretations for style transfer project.
    # save and return
    # made a copy of this
    return m_interpretations, binning_details, motioncodes4vis



################################################################################
## PREPARE INPUT
################################################################################

ALL_JOINT_NAMES2ID = {jn:i for i, jn in enumerate(ALL_JOINT_NAMES)}


def compute_wrist_middle2ndphalanx_distance(coords):
    x = distance_between_joint_pairs([
        [ALL_JOINT_NAMES2ID["left_middle2"], ALL_JOINT_NAMES2ID["left_wrist"]],
        [ALL_JOINT_NAMES2ID["right_middle2"], ALL_JOINT_NAMES2ID["right_wrist"]]], coords)
    return x.mean().item()


def prepare_input(coords):
    """
    Select coordinates for joints of interest, and complete thems with the
    coordinates of virtual joints. If coordinates are provided for the main 22
    joints only, add a prosthesis 2nd phalanx to the middle L&R fingers, in the
    continuity of the forearm.
    
    Args:
        coords (torch.tensor): size (nb of poses, nb of joints, 3), coordinates
            of the different joints, for several poses; with joints being all
            of those defined in ALL_JOINT_NAMES or just the first 22 joints.
    
    Returns:
        (torch.tensor): size (nb of poses, nb of joints, 3), coordinates
            of the different joints, for several poses; with the joints being
            those defined in JOINT_NAMES
    """
    nb_joints = coords.shape[1]
    ### get coords of necessary existing joints
    if nb_joints == 22:
        # add prosthesis phalanxes
        # distance to the wrist
        x = 0.1367 # found by running compute_wrist_middle2ndphalanx_distance on the 52-joint sized coords of a 20k-pose set
        # direction from the wrist (vectors), in the continuity of the forarm
        left_v = coords[:,ALL_JOINT_NAMES2ID["left_wrist"]] - coords[:,ALL_JOINT_NAMES2ID["left_elbow"]]
        right_v = coords[:,ALL_JOINT_NAMES2ID["right_wrist"]] - coords[:,ALL_JOINT_NAMES2ID["right_elbow"]]
        # new phalanx coordinate
        added_j = [x*left_v/torch.linalg.norm(left_v, axis=1).view(-1,1) \
                        + coords[:,ALL_JOINT_NAMES2ID["left_wrist"]],
                    x*right_v/torch.linalg.norm(right_v, axis=1).view(-1,1) \
                        + coords[:,ALL_JOINT_NAMES2ID["right_wrist"]]]
        added_j = [aj.view(-1, 1, 3) for aj in added_j]
        coords = torch.cat([coords] + added_j, axis=1) # concatenate along the joint axis
    if nb_joints >= 52:
        # remove unecessary joints
        keep_joints_indices = [ALL_JOINT_NAMES2ID[jn] for jn in JOINT_NAMES[:-len(VIRTUAL_JOINTS)]]
        coords = coords[:,keep_joints_indices]
    ### add virtual joints
    added_j = [0.5*(coords[:,JOINT_NAMES2ID["left_wrist"]] + coords[:,JOINT_NAMES2ID["left_middle2"]]), # left hand
                0.5*(coords[:,JOINT_NAMES2ID["right_wrist"]] + coords[:,JOINT_NAMES2ID["right_middle2"]]), # right hand
                1/3*(coords[:,JOINT_NAMES2ID["pelvis"]] + coords[:,JOINT_NAMES2ID["neck"]] + coords[:,JOINT_NAMES2ID["spine3"]]), # torso
                # ADD_VIRTUAL_JOINT
                ]
    added_j = [aj.view(-1, 1, 3) for aj in added_j]
    coords = torch.cat([coords] + added_j, axis=1) # concatenate along the joint axis
    return coords


def prepare_posecode_queries():
    """
    Returns a dict with data attached to each kind of posecode, for all
    posecodes of the given kind. One posecode is defined by its kind, joint set
    and interpretation. The joint set does not always carry the name of the body
    part that is actually described by the posecode, and will make it to the
    text. Hence the key 'focus body part'.
    Specifically:
    - the tensor of jointset ids (1 joint set/posecode, with the size of the
        joint set depending on the kind of posecode). The order of the ids might
        matter.
    - the list of acceptable interpretations ids for each jointset (at least 1
        acceptable interpretation/jointset)
    - the list of unskippable interpretations ids for each jointset (possible to
        have empty lists)
    - the list of support-I interpretation ids for each jointset (possible to
        have empty list)
    - the list of support-II interpretation ids for each jointset (possible to
        have empty list)
    - the name of the main focus body part for each jointset
    - the offset to convert the interpretation ids (valid in the scope of the
        considered posecode operator) to global interpretation ids
    """
    posecode_queries = {}
    offset = 0
    for posecode_kind, posecode_list in ALL_ELEMENTARY_POSECODES.items():
        # fill in the blanks for acceptable interpretation (when defining posecodes, '[]' means that all operator interpretation are actually acceptable)
        acceptable_intptt_names = [p[2] if p[2] else POSECODE_OPERATORS_VALUES[posecode_kind]['category_names'] for p in posecode_list]

        print(posecode_kind)

        # parse information about the different posecodes
        joint_ids = torch.tensor([[JOINT_NAMES2ID[jname] for jname in p[0]]
                                    if type(p[0])!=str else JOINT_NAMES2ID[p[0]]
                                    for p in posecode_list]).view(len(posecode_list), -1)
        acceptable_intptt_ids = [[INTPTT_NAME2ID[ain_i] for ain_i in ain]
                                    for ain in acceptable_intptt_names]
        rare_intptt_ids = [[INTPTT_NAME2ID[rin_i] for rin_i in p[3]]
                                    for p in posecode_list]
        support_intptt_ids_typeI = [[INTPTT_NAME2ID[sin_i[0]] for sin_i in p[4] if sin_i[1]==1]
                                    for p in posecode_list]
        support_intptt_ids_typeII = [[INTPTT_NAME2ID[sin_i[0]] for sin_i in p[4] if sin_i[1]==2]
                                    for p in posecode_list]

        # sanity checks
        # - an interpretation cannot be both a rare and a support-I interpretation
        tmp = [len([rin_i for rin_i in rare_intptt_ids[i] if rin_i in support_intptt_ids_typeI[i]]) for i in range(len(posecode_list))]
        if sum(tmp):
            print(f'An interpretation cannot be both a rare and a support interpretation of type I.')
            for t in tmp:
                if t:
                    print(f'Error in definition of posecode {posecode_list[t][0]} [number {t+1} of {posecode_kind} kind].')
            sys.exit()
        # - a posecode should not be defined twice for the same kind of posecode
        unique  = set([tuple(set(jid.tolist())) for jid in joint_ids])
        if len(unique) < len(joint_ids):
            print(f'Error in posecode definition of [{posecode_kind} kind]. A posecode should'
                  f' only be defined once. Check unicity of joint sets (considering involved '
                  f'joints in any order). Change interpretations, as well as the focus body '
                  f'parts if necessary, so that the joint set if used only once for this kind of posecode.')
            sys.exit()

        # save posecode information
        posecode_queries[posecode_kind] = {
            "joint_ids": joint_ids,
            "acceptable_intptt_ids": acceptable_intptt_ids,
            "rare_intptt_ids": rare_intptt_ids,
            "support_intptt_ids_typeI": support_intptt_ids_typeI,
            "support_intptt_ids_typeII": support_intptt_ids_typeII,
            "focus_body_part": [p[1] for p in posecode_list],
            "offset": offset,
        }
        offset += len(POSECODE_OPERATORS_VALUES[posecode_kind]['category_names'])
    return posecode_queries


def prepare_motioncode_queries():
    """
    Returns a dict with data attached to each kind of posecode, for all
    posecodes of the given kind. One posecode is defined by its kind, joint set
    and interpretation. The joint set does not always carry the name of the body
    part that is actually described by the posecode, and will make it to the
    text. Hence the key 'focus body part'.
    Specifically:
    - the tensor of jointset ids (1 joint set/posecode, with the size of the
        joint set depending on the kind of posecode). The order of the ids might
        matter.
    - the list of acceptable interpretations ids for each jointset (at least 1
        acceptable interpretation/jointset)
    - the list of unskippable interpretations ids for each jointset (possible to
        have empty lists)
    - the list of support-I interpretation ids for each jointset (possible to
        have empty list)
    - the list of support-II interpretation ids for each jointset (possible to
        have empty list)
    - the name of the main focus body part for each jointset
    - the offset to convert the interpretation ids (valid in the scope of the
        considered posecode operator) to global interpretation ids
    """
    motioncode_queries = {}
    offset = 0
    for motioncode_kind, motioncode_list in ALL_ELEMENTARY_MOTIONCODES.items():
        # fill in the blanks for acceptable interpretation (when defining posecodes, '[]' means that all operator interpretation are actually acceptable)
        spatial_acceptable_intptt_names = [p[2] if p[2] else MOTIONCODE_OPERATORS_VALUES[motioncode_kind]['category_names'] for p in
                                   motioncode_list]
        try:
            temporal_acceptable_intptt_names = []
            for p in motioncode_list:
                if p[5] == None: # This means there is no eligible intp. Used by spatial relation motioncodes
                    temporal_acceptable_intptt_names.append([])
                elif p[5] == []:
                    temporal_acceptable_intptt_names.append(MOTIONCODE_OPERATORS_VALUES[motioncode_kind]['category_names_velocity'])
                else:
                    temporal_acceptable_intptt_names.append(p[5])

            # temporal_acceptable_intptt_names = [p[5] if p[5] else MOTIONCODE_OPERATORS_VALUES[motioncode_kind]['category_names_velocity']
            #                                    for p in
            #                                    motioncode_list]
        except:
            print("!")
        # print(motioncode_kind)

        # parse information about the different posecodes
        joint_ids = torch.tensor([[JOINT_NAMES2ID[jname] for jname in p[0]]
                                  if type(p[0]) != str else JOINT_NAMES2ID[p[0]]
                                  for p in motioncode_list]).view(len(motioncode_list), -1)

        spatial_acceptable_intptt_ids = [[INTPTT_NAME2ID_MOTION[ain_i] for ain_i in ain]
                                 for ain in spatial_acceptable_intptt_names]
        temporal_acceptable_intptt_ids = [[INTPTT_NAME2ID_MOTION[ain_i] for ain_i in ain]
                                          for ain in temporal_acceptable_intptt_names]

        spatial_rare_intptt_ids = [[INTPTT_NAME2ID_MOTION[rin_i] for rin_i in p[3]]
                           for p in motioncode_list]
        try:
            temporal_rare_intptt_ids = [[INTPTT_NAME2ID_MOTION[rin_i] for rin_i in p[6]]
                                        for p in motioncode_list]
        except:
            print()

        spatial_support_intptt_ids_typeI = [[INTPTT_NAME2ID_MOTION[sin_i[0]] for sin_i in p[4] if sin_i[1] == 1]
                                    for p in motioncode_list]
        spatial_support_intptt_ids_typeII = [[INTPTT_NAME2ID_MOTION[sin_i[0]] for sin_i in p[4] if sin_i[1] == 2]
                                     for p in motioncode_list]

        temporal_support_intptt_ids_typeI = [[INTPTT_NAME2ID_MOTION[sin_i[0]] for sin_i in p[4] if sin_i[1] == 1]
                                            for p in motioncode_list]
        temporal_support_intptt_ids_typeII = [[INTPTT_NAME2ID_MOTION[sin_i[0]] for sin_i in p[4] if sin_i[1] == 2]
                                             for p in motioncode_list]

        # sanity checks
        # - an interpretation cannot be both a rare and a support-I interpretation
        for rare_intptt_ids, support_intptt_ids_typeI in zip([spatial_rare_intptt_ids, temporal_rare_intptt_ids, ],
                                                             [spatial_support_intptt_ids_typeI, spatial_support_intptt_ids_typeII]):
            tmp = [len([rin_i for rin_i in rare_intptt_ids[i] if rin_i in support_intptt_ids_typeI[i]]) for i in
                   range(len(motioncode_list))]
            if sum(tmp):
                print(f'An interpretation cannot be both a rare and a support interpretation of type I.')
                for t in tmp:
                    if t:
                        print(
                            f'Error in definition of posecode {motioncode_list[t][0]} [number {t + 1} of {motioncode_kind} kind].')
                sys.exit()


        # - a posecode should not be defined twice for the same kind of posecode
        unique = set([tuple(set(jid.tolist())) for jid in joint_ids])
        if len(unique) < len(joint_ids):
            print(f'Error in posecode definition of [{motioncode_kind} kind]. A posecode should'
                  f' only be defined once. Check unicity of joint sets (considering involved '
                  f'joints in any order). Change interpretations, as well as the focus body '
                  f'parts if necessary, so that the joint set if used only once for this kind of posecode.')
            sys.exit()

        # save posecode information
        motioncode_queries[motioncode_kind] = {
            "joint_ids": joint_ids,

            # Spatial
            "spatial_acceptable_intptt_ids": spatial_acceptable_intptt_ids,
            "spatial_rare_intptt_ids": spatial_rare_intptt_ids,
            "spatial_support_intptt_ids_typeI": spatial_support_intptt_ids_typeI,
            "spatial_support_intptt_ids_typeII": spatial_support_intptt_ids_typeII,
            # Temporal
            "temporal_acceptable_intptt_ids": temporal_acceptable_intptt_ids,
            "temporal_rare_intptt_ids": temporal_rare_intptt_ids,
            "temporal_support_intptt_ids_typeI": temporal_support_intptt_ids_typeI,
            "temporal_support_intptt_ids_typeII": temporal_support_intptt_ids_typeII,

            "focus_body_part": [p[1] for p in motioncode_list],
            "offset": offset,
        }
        offset += len(MOTIONCODE_OPERATORS_VALUES[motioncode_kind]['category_names'])
    return motioncode_queries

def prepare_super_posecode_queries(p_queries):
    """
    Returns a dict with data attached to each super-posecode (represented by
    their super-posecode ID):
    - the list of different ways to produce the super-posecode, with each way
        being a sublist of required posecodes, and each required posecode is
        representaed by a list of size 3, with:
        - their kind
        - the index of the column in the matrix of elementary posecode
          interpretation (which is specific to the posecode kind) to look at (ie.
          the index of posecode in the posecode list of the corresponding kind)
        - the expected interpretation id to search in this column
    - a boolean indicating whether this is a rare posecode
    - the interpretation id of the super-posecode
    - the focus body part name for the super-posecode
    """
    super_posecode_queries = {}
    for sp in SUPER_POSECODES:
        sp_id = sp[0]
        required_posecodes = []
        # iterate over the ways to produce the posecode
        for w in SUPER_POSECODES_REQUIREMENTS[sp_id]:
            # iterate over required posecodes
            w_info = []
            for req_p in w:
                # req_p[0] is the kind of elementary posecode
                # req_p[1] is the joint set of the elementary posecode
                # req_p[2] is the required interpretation for the elementary
                # posecode
                # Basically, the goal is to convert everything into ids. As the
                # joint set is the one of an existing posecode, it will be
                # represented by the index of the posecode instead of the tensor
                # of the joint ids.
                # 1) convert joint names to joint ids
                # req_p_js = torch.tensor([JOINT_NAMES2ID[jname] for jname in req_p[1]])
                req_p_js = torch.tensor([JOINT_NAMES2ID[jname] for jname in req_p[1]]
                                    if type(req_p[1])!=str else [JOINT_NAMES2ID[req_p[1]]]).view(1,-1)
                # 2) search for the index of the posecode represented by this
                # joint set in the list of posecodes of the corresponding kind
                # NOTE: this joint set is supposed to be unique (see function
                # prepare_posecode_queries)
                try:
                    req_p_ind = torch.where((p_queries[req_p[0]]['joint_ids'] == req_p_js).all(1))[0][0].item()
                except IndexError:
                    print(f"Elementary posecode {req_p} is used for a super-posecode but seems not to be defined.")
                    sys.exit()
                # 3) convert the interpretation to an id, and
                # 4) add the posecode requirement to the list thereof
                w_info.append([req_p[0], req_p_ind, INTPTT_NAME2ID[req_p[2]]])
            required_posecodes.append(w_info)
        # save super-posecode information
        super_posecode_queries[sp_id] = {
            "required_posecodes":required_posecodes,
            "is_rare": sp[2],
            "intptt_id": INTPTT_NAME2ID[sp[1][1]],
            "focus_body_part": sp[1][0]
        }
    return super_posecode_queries


################################################################################
## INFER POSECODES
################################################################################

def infer_posecodes(coords, p_queries, sp_queries,
                    request='Low_Level_intptt', verbose = True):
    
    # init
    # init
    nb_poses = len(coords)
    p_interpretations = {}
    p_eligibility = {}

    for p_kind, p_operator in POSECODE_OPERATORS.items():
        # evaluate posecodes
        val = p_operator.eval(p_queries[p_kind]["joint_ids"], coords)

        # This happens in the case of only one frame
        # if len(val.shape) == 1: # != (1, p_queries[p_kind]["joint_ids"].shape[0]):
        #     print() # val = val.unsqueeze(0)

        # to represent a bit human subjectivity, slightly randomize the
        # thresholds, or, more conveniently, simply randomize a bit the
        # evaluations: add or subtract up to the maximum authorized random
        # offset to the measured values.
        # q = (torch.rand(val.shape)*2-1) * p_operator.random_max_offset
        # a, b = (torch.rand(val.shape) * 2 - 1) , p_operator.random_max_offset
        val += (torch.rand(val.shape)*2-1).to(device) * p_operator.random_max_offset
        # interprete the measured values
        p_intptt = p_operator.interprete(val) + p_queries[p_kind]["offset"]
        # infer posecode eligibility for description
        p_elig = torch.zeros(p_intptt.shape)
        try:
            for js in range(p_intptt.shape[1]):
                print()
        except:
            print()

        for js in range(p_intptt.shape[1]): # nb of joint sets --> it is infact equal to the number of queries
            intptt_a = torch.tensor(p_queries[p_kind]["acceptable_intptt_ids"][js])
            intptt_r = torch.tensor(p_queries[p_kind]["rare_intptt_ids"][js])
            # * fill with 1 if the measured interpretation is one of the
            #   acceptable ones,
            # * fill with 2 if, in addition, it is one of the nonskippables
            #   ones,
            # * fill with 0 otherwise
            # * Note that support interpretations are necessarily acceptable
            #   interpretations (otherwise they would not make it to the
            #   super-posecode inference step); however depending on the
            #   support-type of the interpretation, the eligibility could be
            #   changed in the future
            aA= p_intptt[:, js].view(-1, 1)
            BB = p_intptt[:, js].view(-1, 1) == intptt_a
            CC= (p_intptt[:, js].view(-1, 1) == intptt_a).sum(1)
            p_elig[:, js] = (p_intptt[:, js].view(-1, 1) == intptt_a).sum(1) + (p_intptt[:, js].view(-1, 1) == intptt_r).sum(1)
        # store values
        p_interpretations[p_kind] = p_intptt  # size (nb of poses, nb of joint sets)
        p_eligibility[p_kind] = p_elig  # size (nb of poses, nb of joint sets)
    
    # if request=='Low_Level_intptt':
    #     return p_interpretations, p_eligibility
    
    
    # Infer super-posecodes from elementary posecodes
    # (this treatment is pose-specific)
    sp_elig = torch.zeros(nb_poses, len(sp_queries))
    for sp_ind, sp_id in enumerate(sp_queries):
        # iterate over the different ways to produce the super-posecode
        for w in sp_queries[sp_id]["required_posecodes"]:
            # check if all the conditions on the elementary posecodes are met
            sp_col = torch.ones(nb_poses)
            for ep in w: # ep = (kind, joint_set_column, intptt_id) for the given elementary posecode
                sp_col = torch.logical_and(sp_col, (p_interpretations[ep[0]][:,ep[1]] == ep[2]))
            # all the ways to produce the super-posecodes must be compatible
            # (ie. no overwriting, one sucessful way is enough to produce the 
            # super-posecode for a given pose)
            sp_elig[:,sp_ind] = torch.logical_or(sp_elig[:,sp_ind], sp_col.view(-1))
        # specify if it is a rare super-posecode
        if sp_queries[sp_id]["is_rare"]:
            sp_elig[:,sp_ind] *= 2
        
    # Treat eligibility for support-I & support-II posecode interpretations This
    # must happen in a second double-loop since we need to know if the
    # super-posecode could be produced in any way beforehand ; and because some
    # of such interpretations can contribute to several distinct superposecodes
    for sp_ind, sp_id in enumerate(sp_queries):
        for w in sp_queries[sp_id]["required_posecodes"]:
            for ep in w: # ep = (kind, joint_set_column, intptt_id) for the given elementary posecode
                # support-I
                if ep[2] in p_queries[ep[0]]["support_intptt_ids_typeI"][ep[1]]:
                    # eligibility set to 0, independently of whether the super-
                    # posecode could be produced or not
                    selected_poses = (p_interpretations[ep[0]][:,ep[1]] == ep[2])
                # support-II
                elif ep[2] in p_queries[ep[0]]["support_intptt_ids_typeII"][ep[1]]:
                    # eligibility set to 0 if the super-posecode production
                    # succeeded (no matter how, provided that the support-II
                    # posecode interpretation was the required one in some other
                    # possible production recipe for the given super-posecode)
                    selected_poses = torch.logical_and(sp_elig[:, sp_ind], (p_interpretations[ep[0]][:,ep[1]] == ep[2]))
                else:
                    # this posecode interpretation is not a support one
                    # its eligibility must not change
                    continue
                p_eligibility[ep[0]][selected_poses, ep[1]] = 0

    # Add super-posecodes as a new kind of posecodes
    p_eligibility["superPosecodes"] = sp_elig

    # We have some supports type I which are meant to be used by active
    # joint (subject) detection algorithm. Therefore, they are support
    # type-I although they are not appeared in the super-posecodes'
    # requirement list. So, we do another round of polishing them.
    for p_kind in p_eligibility:
        if p_kind=='superPosecodes': continue
        p_intptt = p_interpretations[p_kind]
        for js in range(p_intptt.shape[1]):  # nb of joint sets --> it is infact equal to the number of querries
            intptt_a = torch.tensor(p_queries[p_kind]["acceptable_intptt_ids"][js])
            support_I = torch.tensor(p_queries[p_kind]["support_intptt_ids_typeI"][js])
            mask_array = 1-(p_intptt[:, js].view(-1, 1) == support_I).sum(1)
            p_eligibility[p_kind][:, js] *= mask_array


    # Print information about the number of posecodes
    if verbose:
        total_posecodes = 0
        print("Number of posecodes of each kind:")
        for p_kind, p_elig in p_eligibility.items():
            print(f'- {p_kind}: {p_elig.shape[1]}')
            total_posecodes += p_elig.shape[1]
        print(f'Total: {total_posecodes} posecodes.')

    return p_interpretations, p_eligibility

Motion2Pose_map = {
                    'angular': 'angle',
                    'proximity': 'distance',

                    'spatial_relation_x': 'relativePosX',
                    'spatial_relation_y': 'relativePosY',
                    'spatial_relation_z': 'relativePosZ',

                    'displacement_x': 'position_x',
                    'displacement_y': 'position_y',
                    'displacement_z': 'position_z',

                    'rotation_pitch': 'orientation_pitch',
                    'rotation_roll': 'orientation_roll',
                    'rotation_yaw': 'orientation_yaw'

                    # "root_orientation_x":
                    # "root_orientation_y":
                    # "root_orientation_z":
#                     Add more mapping between positionals and actions
}
def infer_motioncodes(coords, p_interpretations, p_queries, sp_queries, m_queries,
                      request='Low_Level_intptt', verbose=True):

    # init
    nb_poses = len(coords)
    m_interpretations = {}
    m_eligibility = {}

    for m_kind, m_operator in MOTIONCODE_OPERATORS.items():

        # First we need to find out the correspondence between js_ids of posecode and motioncode joint sets
        # Otherwise they should be definded exactly in a same oreder in queries. Also, if the number of posecodes
        # and motioncodes for a specific kind like position_x and displacement_x are not equal, it doesn't work
        p_m_js_ids = []
        for mjs_id in range(m_queries[m_kind]["joint_ids"].shape[0]):
            m_js = m_queries[m_kind]["joint_ids"][mjs_id]
            matching_rows = torch.all(p_queries[Motion2Pose_map[m_kind]]['joint_ids'] == m_js, dim=1)
            pj_id = torch.where(matching_rows)[0].cpu().numpy().item()
            p_m_js_ids.append ({'m_js': m_js, 'mjs_id': mjs_id, 'pj_id':pj_id})

        # evaluate motioncodes based on posecodes i.e. motion-detection
        val = m_operator.eval(p_m_js_ids, coords, p_interpretations[Motion2Pose_map[m_kind]]) # 'angular' to 'angle'

        # to represent a bit human subjectivity, slightly randomize the
        # thresholds, or, more conveniently, simply randomize a bit the
        # evaluations: add or subtract up to the maximum authorized random
        # offset to the measured values.
        # q = (torch.rand(val.shape) * 2 - 1) * m_operator.random_max_offset
        # a, b = (torch.rand(val.shape) * 2 - 1), m_operator.random_max_offset
        # val += (torch.rand(val.shape) * 2 - 1) * m_operator.random_max_offset

        # Because it is already added at the posecode infer step.

        # interprete the measured values
        m_intptt = m_operator.interprete(val) #+ m_queries[m_kind]["offset"]

        # Now we need to add offset to all interprations
        for i_intpt in range(len(m_intptt)): # iterate over joint sets
            if m_intptt[i_intpt] == [] : continue
            for j_intpt in range(len(m_intptt[i_intpt])): # iterate over detected motions
                # We do this since tuple object does not support item assignment
                # m_intptt[i_intpt][j_intpt]=(m_intptt[i_intpt][j_intpt][0] + m_queries[m_kind]["offset"],    # Spatial
                #                             m_intptt[i_intpt][j_intpt][1] + GLOBAL_VELOCITY_INTTPT_OFFSET) # Temporal


                m_intptt[i_intpt][j_intpt]['spatial'] +=  m_queries[m_kind]["offset"]
                m_intptt[i_intpt][j_intpt]['temporal'] += GLOBAL_VELOCITY_INTTPT_OFFSET
                m_intptt[i_intpt][j_intpt]['start'] += 0
                m_intptt[i_intpt][j_intpt]['end'] += 0
                m_intptt[i_intpt][j_intpt]['end'] += 0
                m_intptt[i_intpt][j_intpt]['posecode'] = [Motion2Pose_map[m_kind], p_m_js_ids[i_intpt]['pj_id']] # we want to keep it for the satartng/ending point

        # infer posecode eligibility for description
        m_elig = torch.zeros(len(m_intptt)) # number of joint sets
        m_elig = [ [] for i in range(len(m_intptt))]

        # * fill with 1 if the measured interpretation is one of the
        #   acceptable ones,
        # * fill with 2 if, in addition, it is one of the nonskippables
        #   ones,
        # * fill with 0 otherwise
        # * Note that support interpretations are necessarily acceptable
        #   interpretations (otherwise they would not make it to the
        #   super-posecode inference step); however depending on the
        #   support-type of the interpretation, the eligibility could be
        #   changed in the future
        for js in range(len(m_intptt)):

            spatial_intptt_a = torch.tensor(m_queries[m_kind]["spatial_acceptable_intptt_ids"][js])
            spatial_intptt_r = torch.tensor(m_queries[m_kind]["spatial_rare_intptt_ids"][js])

            temporal_intptt_a = torch.tensor(m_queries[m_kind]["temporal_acceptable_intptt_ids"][js])
            temporal_intptt_r = torch.tensor(m_queries[m_kind]["temporal_rare_intptt_ids"][js])

            for intensity_velocity in m_intptt[js]:
                # e_intensity = (intensity_velocity[0] == spatial_intptt_a).sum(0) + (intensity_velocity[0] == spatial_intptt_r).sum(0)
                # e_velocity = (intensity_velocity[1] == temporal_intptt_a).sum(0) + (intensity_velocity[1] == temporal_intptt_r).sum(0)
                # m_elig[js].append( (e_intensity, e_velocity) )
                e_intensity = (intensity_velocity['spatial'] == spatial_intptt_a).sum(0) + (
                            intensity_velocity['spatial'] == spatial_intptt_r).sum(0)
                e_velocity = (intensity_velocity['temporal'] == temporal_intptt_a).sum(0) + (
                            intensity_velocity['temporal'] == temporal_intptt_r).sum(0)
                m_elig[js].append((e_intensity, e_velocity)) # we assume it was enough to carry start and end info. with the m_intptt, no need to add here


        # store values
        m_interpretations[m_kind] = m_intptt  # size (nb of poses, nb of joint sets)
        m_eligibility[m_kind] = m_elig  # size (nb of poses, nb of joint sets)

    if request == 'Low_Level_intptt':
        return m_interpretations, m_eligibility

    # Infer super-posecodes from elementary posecodes
    # (this treatment is pose-specific)
    sp_elig = torch.zeros(nb_poses, len(sp_queries))
    for sp_ind, sp_id in enumerate(sp_queries):
        # iterate over the different ways to produce the super-posecode
        for w in sp_queries[sp_id]["required_posecodes"]:
            # check if all the conditions on the elementary posecodes are met
            sp_col = torch.ones(nb_poses)
            for ep in w:  # ep = (kind, joint_set_column, intptt_id) for the given elementary posecode
                sp_col = torch.logical_and(sp_col, (p_interpretations[ep[0]][:, ep[1]] == ep[2]))
            # all the ways to produce the super-posecodes must be compatible
            # (ie. no overwriting, one sucessful way is enough to produce the
            # super-posecode for a given pose)
            sp_elig[:, sp_ind] = torch.logical_or(sp_elig[:, sp_ind], sp_col.view(-1))
        # specify if it is a rare super-posecode
        if sp_queries[sp_id]["is_rare"]:
            sp_elig[:, sp_ind] *= 2

    # Treat eligibility for support-I & support-II posecode interpretations This
    # must happen in a second double-loop since we need to know if the
    # super-posecode could be produced in any way beforehand ; and because some
    # of such interpretations can contribute to several distinct superposecodes
    for sp_ind, sp_id in enumerate(sp_queries):
        for w in sp_queries[sp_id]["required_posecodes"]:
            for ep in w:  # ep = (kind, joint_set_column, intptt_id) for the given elementary posecode
                # support-I
                if ep[2] in p_queries[ep[0]]["support_intptt_ids_typeI"][ep[1]]:
                    # eligibility set to 0, independently of whether the super-
                    # posecode could be produced or not
                    selected_poses = (p_interpretations[ep[0]][:, ep[1]] == ep[2])
                # support-II
                elif ep[2] in p_queries[ep[0]]["support_intptt_ids_typeII"][ep[1]]:
                    # eligibility set to 0 if the super-posecode production
                    # succeeded (no matter how, provided that the support-II
                    # posecode interpretation was the required one in some other
                    # possible production recipe for the given super-posecode)
                    selected_poses = torch.logical_and(sp_elig[:, sp_ind],
                                                       (p_interpretations[ep[0]][:, ep[1]] == ep[2]))
                else:
                    # this posecode interpretation is not a support one
                    # its eligibility must not change
                    continue
                p_eligibility[ep[0]][selected_poses, ep[1]] = 0

    # Add super-posecodes as a new kind of posecodes
    p_eligibility["superPosecodes"] = sp_elig

    # Print information about the number of posecodes
    if verbose:
        total_posecodes = 0
        print("Number of posecodes of each kind:")
        for p_kind, p_elig in p_eligibility.items():
            print(f'- {p_kind}: {p_elig.shape[1]}')
            total_posecodes += p_elig.shape[1]
        print(f'Total: {total_posecodes} posecodes.')

    return p_interpretations, p_eligibility


################################################################################
## FORMAT POSECODES
################################################################################

def parse_joint(joint_name):
    # returns side, body_part
    x = joint_name.split("_")
    return x if len(x) == 2 else [None] + x


def parse_super_posecode_joints(sp_id, sp_queries):
    # only a focus body part
    side_1, body_part_1 = parse_joint(sp_queries[sp_id]['focus_body_part'])
    return side_1, body_part_1, None, None


def parse_posecode_joints(p_ind, p_kind, p_queries):
    # get the side & body part of the joints involved in the posecode
    focus_joint = p_queries[p_kind]['focus_body_part'][p_ind]
    # first (main) joint
    if focus_joint is None:
        # no main joint is defined
        bp1_name = JOINT_NAMES[p_queries[p_kind]['joint_ids'][p_ind][0]] # first joint
        side_1, body_part_1 = parse_joint(bp1_name)
    else:
        side_1, body_part_1 = parse_joint(focus_joint)
    # second (support) joint
    if p_kind in POSECODE_KIND_FOCUS_JOINT_BASED or p_kind in MOTIONCODE_KIND_FOCUS_JOINT_BASED: # <added to handle motioncodes>
        # no second joint involved
        side_2, body_part_2 = None, None
    else:
        # print(p_kind, p_ind)
        # q = p_queries[p_kind]['joint_ids'][p_ind]
        bp2_name = JOINT_NAMES[p_queries[p_kind]['joint_ids'][p_ind][1]] # second joint
        side_2, body_part_2 = parse_joint(bp2_name)
    return side_1, body_part_1, side_2, body_part_2


def add_posecode(data, skipped, p, p_elig_val, random_skip, nb_skipped,
                side_1, body_part_1, side_2, body_part_2, intptt_id,
                posecode_address,
                extra_verbose=False):
    # always consider rare posecodes (p_elig_val=2),
    # and randomly ignore skippable ones, up to PROP_SKIP_POSECODES,
    # if applying random skip
    if (p_elig_val == 2) or \
        (p_elig_val and (not random_skip or random.random() >= PROP_SKIP_POSECODES)):
        data[p].append([side_1, body_part_1, intptt_id, side_2, body_part_2, [posecode_address] ]) # deal with interpretation ids for now
        if extra_verbose and p_elig_val == 2: print("NON SKIPPABLE", data[p][-1])
    elif random_skip and p_elig_val:
        skipped[p].append([side_1, body_part_1, intptt_id, side_2, body_part_2, [posecode_address] ])
        nb_skipped += 1
        if extra_verbose: print("skipped", [side_1, body_part_1, intptt_id, side_2, body_part_2])
    return data, skipped, nb_skipped


def add_motioncode(data, skipped, skipped_just_temporal, m, m_elig_val, random_skip, nb_skipped,
                side_1, body_part_1, side_2, body_part_2, intptt_id,
                mc_info, extra_verbose=False):
    # always consider rare posecodes (p_elig_val=2),
    # and randomly ignore skippable ones, up to PROP_SKIP_POSECODES,
    # if applying random skip

    # Note that, we always consider spatial eligibility as the main eligibility
    # Later, if we pick a motion, we decide about the velocity

    # Spatial aspect
    if (m_elig_val[0] == 2) or \
        (m_elig_val[0] and (not random_skip or random.random() >= PROP_SKIP_MOTONCODES_SPATIAL)):

        # Temporal aspect: Note that, we definitely will append this motioncode
        # to the list since it has passed the spatial condition, however, it is
        # necessary to check the temporal condition. That's why we also have an
        # "else" condition in addition to the spatial "if" and "elif".
        # We might combine two latter conditions.

        if (m_elig_val[1] == 2) or \
                (m_elig_val[1] and (not random_skip or random.random() >= PROP_SKIP_MOTONCODES_TEMPORAL)):
            # data.append([side_1, body_part_1, (intptt_id[0], intptt_id[1]), side_2, body_part_2]) # deal with interpretation ids for now
            data.append([side_1, body_part_1,
                         {'spatial': intptt_id['spatial'],
                          'temporal': intptt_id['temporal'],
                          'start': intptt_id['start'],
                          'end': intptt_id['end'],
                          'posecode': [tuple(intptt_id['posecode'])],
                          'mc_info': mc_info
                          },
                         side_2, body_part_2])  # deal with interpretation ids for now

        elif random_skip and m_elig_val[1]:
            skipped_just_temporal.append([side_1, body_part_1, intptt_id, side_2, body_part_2])
            # data.append([side_1, body_part_1, (intptt_id[0],None), side_2, body_part_2])  # deal with interpretation ids for now
            data.append([side_1, body_part_1,
                         {'spatial': intptt_id['spatial'],
                          'temporal': None,
                          'start': intptt_id['start'],
                          'end': intptt_id['end'],
                          'posecode': [tuple(intptt_id['posecode'])],
                          'mc_info': mc_info
                          },
                         side_2, body_part_2])  # deal with interpretation ids for now

        else:
            # data.append([side_1, body_part_1, (intptt_id[0], None), side_2, body_part_2])  # deal with interpretation ids for now
            data.append([side_1, body_part_1,
                         {'spatial': intptt_id['spatial'],
                          'temporal': None,
                          'start': intptt_id['start'],
                          'end': intptt_id['end'],
                          'posecode': [tuple(intptt_id['posecode'])],
                          'mc_info': mc_info
                          },
                         side_2, body_part_2])  # deal with interpretation ids for now



        if extra_verbose and m_elig_val == 2: print("NON SKIPPABLE", data[p][-1])



    elif random_skip and m_elig_val[0]:
        skipped.append([side_1, body_part_1, intptt_id, side_2, body_part_2])
        nb_skipped += 1
        if extra_verbose: print("skipped", [side_1, body_part_1, intptt_id, side_2, body_part_2])



    return data, skipped, skipped_just_temporal, nb_skipped

def format_and_skip_posecodes(p_interpretations, p_eligibility, p_queries, sp_queries,
                                random_skip, verbose=True, extra_verbose=False):
    """
    From classification matrices of the posecodes to a (sparser) data structure.

    Args:
        p_eligibility: dictionary, containing an eligibility matrix per kind
            of posecode. Eligibility matrices are of size (nb of poses, nb of
            posecodes), and contain the following values:
            - 1 if the posecode interpretation is one of the acceptable ones,
            - 2 if, in addition, it is one of the rare (unskippable) ones,
            - 0 otherwise

    Returns:
        2 lists containing a sublist of posecodes for each pose.
        Posecodes are represented as lists of size 5:
        [side_1, body_part_1, intptt_id, side_2, body_part_2]
        The first list is the list of posecodes that should make it to the
        description. The second list is the list of skipped posecodes.
    """

    nb_poses = len(p_interpretations[list(p_interpretations.keys())[0]])
    data = [[] for i in range(nb_poses)] # posecodes that will make it to the description
    skipped = [[] for i in range(nb_poses)] # posecodes that will be skipped
    nb_eligible = 0
    nb_nonskippable = 0
    nb_skipped = 0

    # parse posecodes
    for p_kind in p_interpretations:
        p_intptt = p_interpretations[p_kind]
        p_elig = p_eligibility[p_kind]
        nb_eligible += (p_elig>0).sum().item()
        nb_nonskippable += (p_elig==2).sum().item()
        for pc in range(p_intptt.shape[1]): # iterate over posecodes
            # get the side & body part of the joints involved in the posecode
            side_1, body_part_1, side_2, body_part_2 = parse_posecode_joints(pc, p_kind, p_queries)
            # format eligible posecodes
            for p in range(nb_poses): # iterate over poses frames
                data, skipped, nb_skipped = add_posecode(data, skipped, p,
                                                p_elig[p, pc],
                                                random_skip, nb_skipped,
                                                side_1, body_part_1,
                                                side_2, body_part_2,
                                                p_intptt[p, pc].item(),
                                                posecode_address=(p_kind, pc), # to keep track of posecode
                                                extra_verbose=extra_verbose)

    # parse super-posecodes (only defined through the eligibility matrix)
    sp_elig = p_eligibility['superPosecodes']
    nb_eligible += (sp_elig>0).sum().item()
    nb_nonskippable += (sp_elig==2).sum().item()
    for sp_ind, sp_id in enumerate(sp_queries): # iterate over super-posecodes
        side_1, body_part_1, side_2, body_part_2  = parse_super_posecode_joints(sp_id, sp_queries)
        for p in range(nb_poses):
            data, skipped, nb_skipped = add_posecode(data, skipped, p,
                                            sp_elig[p, sp_ind],
                                            random_skip, nb_skipped,
                                            side_1, body_part_1,
                                            side_2, body_part_2,
                                            sp_queries[sp_id]["intptt_id"],
                                            posecode_address=('Super_posecode', sp_id),
                                            extra_verbose=extra_verbose)

    # check if there are poses with no posecodes, and fix them if possible
    nb_empty_description = 0
    nb_fixed_description = 0
    for p in range(nb_poses):
        if len(data[p]) == 0:
            nb_empty_description += 1
            if not skipped[p]:
                if extra_verbose:
                    # just no posecode available (as none were skipped)
                    print("No eligible posecode for pose {}.".format(p))
            elif random_skip:
                # if some posecodes were skipped earlier, use them for pose
                # description to avoid empty descriptions
                data[p].extend(skipped[p])
                nb_skipped -= len(skipped[p])
                skipped[p] = []
                nb_fixed_description += 1

    if verbose:
        print(f"Total number of eligible posecodes: {nb_eligible} (shared over {nb_poses} poses).")
        print(f"Total number of skipped posecodes: {nb_skipped} (non-skippable: {nb_nonskippable}).")
        print(f"Found {nb_empty_description} empty descriptions.")
        if nb_empty_description > 0:
            print(f"Fixed {round(nb_fixed_description/nb_empty_description*100,2)}% ({nb_fixed_description}/{nb_empty_description}) empty descriptions by considering all eligible posecodes (no skipping).")

    return data, skipped

#############################MOTION#####################
def format_and_skip_motioncodes(m_interpretations, m_eligibility, m_queries, sm_queries,
                                random_skip, verbose=True, extra_verbose=False):
    """
    From classification matrices of the posecodes to a (sparser) data structure.

    Args:
        p_eligibility: dictionary, containing an eligibility matrix per kind
            of posecode. Eligibility matrices are of size (nb of poses, nb of
            posecodes), and contain the following values:
            - 1 if the posecode interpretation is one of the acceptable ones,
            - 2 if, in addition, it is one of the rare (unskippable) ones,
            - 0 otherwise

    Returns:
        2 lists containing a sublist of posecodes for each pose.
        Posecodes are represented as lists of size 5:
        [side_1, body_part_1, intptt_id, side_2, body_part_2]
        The first list is the list of posecodes that should make it to the
        description. The second list is the list of skipped posecodes.
    """

    nb_poses = len(m_interpretations[list(m_interpretations.keys())[0]])
    data = [[] for i in range(nb_poses)] # posecodes that will make it to the description  Todo: fix this for motion.
    skipped = [[] for i in range(nb_poses)] # posecodes that will be skippe
    data, skipped = [], []
    skipped_just_temporal = []

    # Spatial
    nb_eligible_spatial = 0
    nb_nonskippable_spatial = 0
    nb_skipped_spatial = 0
    # Temporal
    nb_eligible_temporal = 0
    nb_nonskippable_temporal = 0
    nb_skipped_temporal = 0

    nb_skipped = 0

    # parse posecodes
    for m_kind in m_interpretations:
        m_intptt = m_interpretations[m_kind]
        m_elig = m_eligibility[m_kind]
        # nb_eligible += (m_elig>0).sum().item()
        # nb_nonskippable += (m_elig==2).sum().item()
        # Counting eligibility from detected motions which might be different
        # from motion code to motion code regarding corresponding pose codes.
        for i in range(len(m_elig)):
            if m_elig[i] == []: continue

            spatial_elig = torch.stack([x[0] for x in m_elig[i]]) # produce an array of the first element
            nb_eligible_spatial += (spatial_elig > 0).sum().item()
            nb_nonskippable_spatial += (spatial_elig == 2).sum().item()

            temporal_elig = torch.stack([x[1] for x in m_elig[i]]) # produce an array of the second element
            nb_eligible_temporal += (temporal_elig > 0).sum().item()
            nb_nonskippable_temporal += (temporal_elig == 2).sum().item()

        for mc in range(len(m_intptt)): # iterate over motioncodes (based on posecodes)  // len because it's a list now
            if m_intptt[mc] == []: continue
            # get the side & body part of the joints involved in the posecode
            side_1, body_part_1, side_2, body_part_2 = parse_posecode_joints(mc, m_kind, m_queries)
            # format eligible posecodes
            # for p in range(nb_poses): # iterate over poses frames
            for m in range(len(m_intptt[mc])): # iterate over detected motions.
                data, skipped, skipped_just_temporal, nb_skipped = add_motioncode(data, skipped, skipped_just_temporal, m,
                                                m_elig[mc][m],# m_elig[m, mc], since it's not an array
                                                              # anymore and it's a list we can't use tuples
                                                random_skip, nb_skipped,
                                                side_1, body_part_1,
                                                side_2, body_part_2,
                                                m_intptt[mc][m], # m_intptt[m, mc].item(), same as m_elig
                                                mc_info={'m_kind': m_kind, 'mc_index': mc,      # We use this info at the agg-->subject-detection
                                                         'focus_body_part': m_queries[m_kind]['focus_body_part'][mc]},
                                                extra_verbose=extra_verbose)

    return data, skipped

    # parse super-posecodes (only defined through the eligibility matrix)
    sp_elig = p_eligibility['superPosecodes']
    nb_eligible += (sp_elig>0).sum().item()
    nb_nonskippable += (sp_elig==2).sum().item()
    for sp_ind, sp_id in enumerate(sp_queries): # iterate over super-posecodes
        side_1, body_part_1, side_2, body_part_2  = parse_super_posecode_joints(sp_id, sp_queries)
        for p in range(nb_poses):
            data, skipped, nb_skipped = add_posecode(data, skipped, p,
                                            sp_elig[p, sp_ind],
                                            random_skip, nb_skipped,
                                            side_1, body_part_1,
                                            side_2, body_part_2,
                                            sp_queries[sp_id]["intptt_id"],
                                            extra_verbose)

    # check if there are poses with no posecodes, and fix them if possible
    nb_empty_description = 0
    nb_fixed_description = 0
    for p in range(nb_poses):
        if len(data[p]) == 0:
            nb_empty_description += 1
            if not skipped[p]:
                if extra_verbose:
                    # just no posecode available (as none were skipped)
                    print("No eligible posecode for pose {}.".format(p))
            elif random_skip:
                # if some posecodes were skipped earlier, use them for pose
                # description to avoid empty descriptions
                data[p].extend(skipped[p])
                nb_skipped -= len(skipped[p])
                skipped[p] = []
                nb_fixed_description += 1

    if verbose:
        print(f"Total number of eligible posecodes: {nb_eligible} (shared over {nb_poses} poses).")
        print(f"Total number of skipped posecodes: {nb_skipped} (non-skippable: {nb_nonskippable}).")
        print(f"Found {nb_empty_description} empty descriptions.")
        if nb_empty_description > 0:
            print(f"Fixed {round(nb_fixed_description/nb_empty_description*100,2)}% ({nb_fixed_description}/{nb_empty_description}) empty descriptions by considering all eligible posecodes (no skipping).")

    return data, skipped





################################################################################
## SELECT POSECODES
################################################################################

# This step is not part of the direct execution of the automatic captioning
# pipeline. It must be executed separately as a preliminary step to determine
# posecodes eligibility (ie. which of them are rare & un-skippable, which are
# not, and which are just too common and trivial to be a description topic).

def superposecode_stats(p_eligibility, sp_queries,
                        prop_eligible=0.4, prop_unskippable=0.06):
    """
    For super-posecodes only.
    Display statistics on posecode interpretations, for the different joint sets.

    Args:
        prop_eligible (float in [0,1]): maximum proportion of poses to which this
            interpretation can be associated for it to me marked as eligible for
            description (ie. acceptable interpretation).
        prop_unskippable (float in [0,1]): maximum proportion of poses to which
            this interpretation can be associated for it to me marked as
            unskippable for description (ie. rare interpretation).
    """
    p_elig = p_eligibility["superPosecodes"]
    nb_poses, nb_sp = p_elig.shape
    results = []
    for sp_ind, sp_id in enumerate(sp_queries): # iterate over super-posecodes
        size = (p_elig[:,sp_ind] > 0).sum().item() / nb_poses # normalize
        verdict = "eligible" if size < prop_eligible else "ignored"
        if size < prop_unskippable:
            verdict = "unskippable"
        results.append([sp_queries[sp_id]['focus_body_part'], INTERPRETATION_SET[sp_queries[sp_id]['intptt_id']], round(size*100, 2), verdict])
    
    # display a nice result table
    print("\n", tabulate(results, headers=["focus body part", "interpretation", "%", "eligibility"]), "\n")


def get_posecode_name(p_ind, p_kind, p_queries):
    """
    Return a displayable 'code' to identify the studied posecode (joint set).
    """
    # get short names for the main & support body parts (if available)
    # NOTE: body_part_1 is always defined
    side_1, body_part_1, side_2, body_part_2 = parse_posecode_joints(p_ind, p_kind, p_queries)
    side_1 = side_1.replace("left", "L").replace("right", "R").replace(PLURAL_KEY, "")+" " if side_1 else ""
    side_2 = side_2.replace("left", "L").replace("right", "R")+" " if side_2 else ""
    body_part_2 = body_part_2 if body_part_2 else ""
    if body_part_1 == body_part_2: # then this is necessarily a sided body part
        tick_text = f'{side_1[:-1]}/{side_2[:-1]} {body_part_1}' # remove "_" in text of side_1/2
    elif side_1 == side_2 and body_part_2: # case of two actual body parts, potentially sided
        if side_1: # sided
            tick_text = f'{side_1[:-1]} {body_part_1}-{body_part_2}' # remove "_" in text of side_1
        else: # not sided
            tick_text = f'{body_part_1} - {body_part_2}' # remove "_" in text of side_1
    else: # different sides
        sbp = f' - {side_2}{body_part_2}' if body_part_2 else ''
        tick_text = f'{side_1}{body_part_1}{sbp}'
    return tick_text.replace("upperarm", "upper arm")


def get_posecode_from_name(p_name):

    # helper function to parse a body part
    def parse_bp(bp):
        if not ("L" in bp or "R" in bp):
            return [None, bp]
        if "L" in bp:
            return ["left", bp.replace("L ", "")]
        else:
            return ["right", bp.replace("R ", "")]

    # parse information about body parts & posecode interpretation
    # (NOTE: one could use the intepretation name instead of interpretation id
    # (by using x.group(3) directly instead of INTPTT_NAME2ID[x.group(3)]) for
    # better portability & understandability, outside of the captioning pipeline)
    x = re.search(r'\[(.*?)\] (.*?) \((.*?)\)', p_name)
    p_kind, bp, intptt = x.group(1), x.group(2), INTPTT_NAME2ID[x.group(3)]

    # depending on the formatting, deduce the body parts at stake
    x = re.search(r'L/R (\w+)', bp)
    if x:
        b = x.group(1)
        return ["left", b, intptt, "right", b]
    
    x = re.search(r'(\w+) (\w+)-(\w+)', bp)
    if x:
        s = x.group(1).replace("L", "left").replace("R", "right")
        return [s, x.group(2), intptt, s, x.group(3)]

    x = re.search(r'([\w\s]+) - ([\w\s]+)', bp)
    # for eg. "L hand - neck" and "L hand - R knee" and "neck - R foot"
    if x:
        bp1, bp2 = parse_bp(x.group(1)), parse_bp(x.group(2))
        return bp1 + [intptt] + bp2

    return parse_bp(bp) + [intptt, None, None]


def get_symmetric_posecode(p):
    if "L/R" in p:
        # just get opposite interpretation
        intptt = re.search(r'\((.*?)\)',p).group(1)
        p = p.replace(intptt, OPPOSITE_CORRESP[intptt])
    else:
        # get symmetric (also works for p="---")
        p = p.replace("L", "$").replace("R", "L").replace("$", "R")
    return p


def posecode_intptt_scatter(p_kind, p_interpretations, p_queries,
                            intptts_names=None, ticks_names=None, title=None,
                            prop_eligible=0.4, prop_unskippable=0.06,
                            jx=0, jy=None, save_fig=False, save_dir="./"):
    """
    For elementary posecodes only.
    Display statistics on posecode interpretations, for the different joint sets.

    Args:
        p_kind (string): kind of posecode to study
        intptts_names (list of strings|None): list of interpretations to study. If
            not specified, all posecode interpretations are studied.
        ticks_names (list of strings|None): displayable interpretation names.
            If not specified, taken from POSECODE_OPERATORS_VALUES.
        title (string|None): title for the created figure
        prop_eligible (float in [0,1]): maximum proportion of poses to which this
            interpretation can be associated for it to me marked as eligible for
            description (ie. acceptable interpretation).
        prop_unskippable (float in [0,1]): maximum proportion of poses to which
            this interpretation can be associated for it to me marked as 
            unskippable for description (ie. rare interpretation).
        jx, jy: define the range of posecodes to be studied (this makes it
            possible to generate diagrams of reasonable size); by default, all
            the posecodes are studied at once.
    """
    p_intptt = p_interpretations[p_kind]
    nb_poses, nb_posecodes = p_intptt.shape
    posecode_range = list(range(jx, jy if jy else nb_posecodes))

    if title != "" and title is None:
        title = f"Statistics for {p_kind} posecodes interpretations ({nb_poses} poses)"
    
    # list of interpretations to study
    ticks_names = ticks_names if ticks_names else POSECODE_OPERATORS_VALUES[p_kind]['category_names_ticks']
    intptts_names = intptts_names if intptts_names else POSECODE_OPERATORS_VALUES[p_kind]['category_names']
    intptt_ids = [INTPTT_NAME2ID[n] for n in intptts_names]
    intptt_ignored_ids = [INTPTT_NAME2ID[n] for n in intptts_names if 'ignored' in n]
    nb_intptts = len(intptt_ids)
    
    # list of joint names to display
    tick_text =  []
    for p_ind in posecode_range: # range(nb_posecodes):
        tick_text.append(get_posecode_name(p_ind, p_kind, p_queries))

    # figure layout
    x = []
    for j in range(len(posecode_range)):
        x += nb_intptts * [j]
    y = list(range(nb_intptts)) * len(posecode_range)
    s = [] # size
    c = [] # color

    # figure data
    for j in posecode_range:
        for ii in intptt_ids:
            size = (p_intptt[:,j] == ii).sum().item() / nb_poses # normalize
            s.append(size * 3000)
            if ii in intptt_ignored_ids:
                c.append('black')
            else:
                if size > prop_eligible:
                    c.append('grey')
                else:
                    c.append('cornflowerblue' if size > prop_unskippable else 'orange')
    
    # set layout
    plt.figure(figsize = (len(posecode_range)*2,nb_intptts))
    offset = 1
    plt.xlim([-offset, len(posecode_range)])
    plt.ylim([-offset, nb_intptts])

    # display data
    plt.scatter(x, y, s, c)
    plt.xticks(np.arange(len(posecode_range)), tick_text, rotation=45, ha="right")
    plt.yticks(np.arange(nb_intptts), ticks_names)
    plt.title(title)

    # save data
    if save_fig:
        save_filepath = os.path.join(save_dir, save_fig)
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        print("Saved figure:", save_filepath)
    plt.show()


def motioncode_intptt_scatter_old(m_kind, m_interpretations, m_queries,
                            intptts_names=None, ticks_names=None, title=None,
                            prop_eligible=0.4, prop_unskippable=0.06,
                            jx=0, jy=None, save_fig=False, save_dir="./"):
    """
    For elementary posecodes only.
    Display statistics on posecode interpretations, for the different joint sets.

    Args:
        p_kind (string): kind of posecode to study
        intptts_names (list of strings|None): list of interpretations to study. If
            not specified, all posecode interpretations are studied.
        ticks_names (list of strings|None): displayable interpretation names.
            If not specified, taken from POSECODE_OPERATORS_VALUES.
        title (string|None): title for the created figure
        prop_eligible (float in [0,1]): maximum proportion of poses to which this
            interpretation can be associated for it to me marked as eligible for
            description (ie. acceptable interpretation).
        prop_unskippable (float in [0,1]): maximum proportion of poses to which
            this interpretation can be associated for it to me marked as
            unskippable for description (ie. rare interpretation).
        jx, jy: define the range of posecodes to be studied (this makes it
            possible to generate diagrams of reasonable size); by default, all
            the posecodes are studied at once.
    """
    m_intptt = m_interpretations[m_kind]
    # nb_poses, nb_posecodes = m_intptt.shape TODO: we should somehow batchify this
    nb_poses = 1
    nb_motioncodes = len(m_intptt)
    motioncode_range = list(range(jx, jy if jy else nb_motioncodes))

    if title != "" and title is None:
        title = f"Statistics for {m_kind} motioncode interpretations ({nb_poses} moions)"

    # list of interpretations to study
    ticks_names = ticks_names if ticks_names else MOTIONCODE_OPERATORS_VALUES[m_kind]['category_names_ticks']
    intptts_names = intptts_names if intptts_names else MOTIONCODE_OPERATORS_VALUES[m_kind]['category_names']
    intptt_ids = [INTPTT_NAME2ID_MOTION[n] for n in intptts_names]
    intptt_ignored_ids = [INTPTT_NAME2ID_MOTION[n] for n in intptts_names if 'ignored' in n]
    nb_intptts = len(intptt_ids)

    # list of joint names to display
    tick_text = []
    for m_ind in motioncode_range:  # range(nb_posecodes):
        tick_text.append(get_posecode_name(m_ind, m_kind, m_queries))

    # figure layout
    x = []
    for j in range(len(motioncode_range)):
        x += nb_intptts * [j]
    y = list(range(nb_intptts)) * len(motioncode_range)
    s = []  # size
    c = []  # color

    # figure data
    for j in motioncode_range:
        for ii in intptt_ids:
            # size = (m_intptt[:, j] == ii).sum().item() / nb_poses  # normalize
            size = sum(1 for item in m_intptt[j] if item.get('spatial') == ii)
            s.append(size * 3000) #????
            if ii in intptt_ignored_ids:
                c.append('black')
            else:
                if size > prop_eligible:
                    c.append('grey')
                else:
                    c.append('cornflowerblue' if size > prop_unskippable else 'orange')

    # set layout
    plt.figure(figsize=(len(motioncode_range) * 2, nb_intptts))
    offset = 1
    plt.xlim([-offset, len(motioncode_range)])
    plt.ylim([-offset, nb_intptts])

    # display data
    plt.scatter(x, y, s, c)
    plt.xticks(np.arange(len(motioncode_range)), tick_text, rotation=45, ha="right")
    plt.yticks(np.arange(nb_intptts), ticks_names)
    plt.title(title)

    # save data
    if save_fig:
        save_filepath = os.path.join(save_dir, save_fig)
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        print("Saved figure:", save_filepath)
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

def motioncode_intptt_scatter(m_kind, m_interpretations, m_queries,
                              intptts_names=None, ticks_names=None, title=None,
                              prop_eligible=0.4, prop_unskippable=0.06,
                              jx=0, jy=None, save_fig=False, save_dir="./"):
    """
    Enhanced visualization for motioncode interpretations scatter plot.
    """
    m_intptt = m_interpretations[m_kind]
    nb_joint_sets = len(m_intptt)
    joint_sets_range = list(range(jx, jy if jy else nb_joint_sets))
    total = sum([len(x) for x in m_intptt])
    if not title:
        title = f"Statistics for {m_kind} motioncode categories distribution"

    ticks_names = ticks_names if ticks_names else MOTIONCODE_OPERATORS_VALUES[m_kind]['category_names_ticks']
    intptts_names = intptts_names if intptts_names else MOTIONCODE_OPERATORS_VALUES[m_kind]['category_names']
    intptt_ids = [INTPTT_NAME2ID_MOTION[n] for n in intptts_names]
    intptt_ignored_ids = [INTPTT_NAME2ID_MOTION[n] for n in intptts_names if 'ignored' in n]
    nb_intptts = len(intptt_ids)

    tick_text = [get_posecode_name(m_ind, m_kind, m_queries) for m_ind in joint_sets_range]

    x = [j for j in range(len(joint_sets_range)) for _ in range(nb_intptts)]
    y = list(range(nb_intptts)) * len(joint_sets_range)
    s = []  # size
    c = []  # color
    ratio_jids_ = []
    for j in joint_sets_range:
        current_joint_set_total = len(m_intptt[j]) if len(m_intptt[j]) else 1
        ratio_jids_.append( current_joint_set_total/total)
        tick_text[j] += f'\n{100*(current_joint_set_total/total):.2f}%'
        for ii in intptt_ids:
            size = sum(1 for item in m_intptt[j] if item.get('spatial') == ii)
            s.append(size/current_joint_set_total)
        # if sum(s):
        #     s = [x/sum(s) for x in s]
        for ii in range(j*len(intptt_ids), (j+1)*len(intptt_ids)):
            if ii in intptt_ignored_ids or s[ii]==0.0:
                color = 'black'
            elif s[ii] > prop_eligible:
                color = 'grey'
            elif  s[ii] > prop_unskippable:
                color = 'cornflowerblue'
            else:
                color = 'orange'
            c.append(color)

    # Normalize sizes
    max_size = max(s)
    min_size = min(s)
    normalized_s = [(size - min_size) / (max_size - min_size) * 3000 + 600 for size in s]

    # Use a colormap
    colormap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=min(s), vmax=max(s))
    # colors = [colormap(norm(size)) for size in s]
    colors = c
    plt.figure(figsize=(len(joint_sets_range) * 2 + 2, nb_intptts + 2))
    plt.scatter(x, y, s=normalized_s, c=colors, alpha=1.0)
    for i, txt in enumerate(s):
        plt.text(x[i], y[i], f"{txt*100:.3f}%", ha='left', va='bottom', fontsize=12, color='Black')

    plt.xticks(np.arange(len(joint_sets_range)), tick_text, rotation=45, ha="right", fontsize=10)
    plt.yticks(np.arange(nb_intptts), ticks_names, fontsize=10)
    plt.title(title)

    # Add a colorbar
    # sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, label='Event Frequency')

    if save_fig:
        # Construct the filename using the `m_kind` parameter and ensure it's saved as a PDF.
        filename = f"{m_kind}_scatter_plot.pdf"
        save_filepath = os.path.join(save_dir, filename)
        plt.savefig(save_filepath, format='pdf', dpi=300, bbox_inches='tight')
        print("Saved figure:", save_filepath)

    plt.show()

    plt.show()

################################################################################
## AGGREGATE POSECODES
################################################################################

def quick_posecode_display(p):
    if p: return p[:2]+[INTERPRETATION_SET[p[2]]]+p[3:]

def same_posecode_family(pA, pB):
    # check if posecodes pA and pB have similar or opposite interpretations
    return pA[2] == pB[2] or (OPPOSITE_CORRESP_ID.get(pB[2], False) and pA[2] == OPPOSITE_CORRESP_ID[pB[2]])

def reverse_joint_order(pA):
    # the first joint becomes the second joint (and vice versa), the
    # interpretation is converted to its opposite
    # (assumes that pA is of size 5)
    return pA[3:] + [OPPOSITE_CORRESP_ID[pA[2]]] + pA[:2]

def pluralize(body_part):
    return PLURALIZE.get(body_part, f"{body_part}s")

def aggregate_posecodes(posecodes, simplified_captions=False,
                        apply_transrel_ripple_effect=True, apply_stat_ripple_effect=True,
                        extra_verbose=False):

    # augment ripple effect rules to have rules for the R side as well
    # (rules registered in the captioning_data were found to apply for L & R,
    # but were registered for the L side only for declaration simplicity)
    stat_rer = STAT_BASED_RIPPLE_EFFECT_RULES + [[get_symmetric_posecode(pc) for pc in l] for l in STAT_BASED_RIPPLE_EFFECT_RULES]
    # format ripple effect rules for later processing
    stat_rer = [[get_posecode_from_name(pc) if pc!="---" else None for pc in l] for l in stat_rer]
    # get stats over posecode discarding based on application of ripple effect rules (rer)
    stat_rer_removed = 0 # rules based on statistically frequent pairs and triplets of posecodes
    transrel_rer_removed = 0 # rules based on transitive relations between body parts
    
    # treat each pose one by one
    nb_poses = len(posecodes)
    for p in range(nb_poses):
        updated_posecodes = copy.deepcopy(posecodes[p])
        
        if extra_verbose: 
            print(f"\n**POSE {p}")
            print("Initial posecodes:")
            print(updated_posecodes)

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 0) Remove redundant information (ripple effect #1)
        # If we have 3 posecodes telling that a < b, b < c and a < c (with 'a',
        # 'b', and 'c' being arbitrary body parts, and '<' representing an order
        # relation such as 'below'), then keep only the posecodes telling that a
        # < b and b < c, as it is enough to infer the global relation a < b < c.
        if apply_transrel_ripple_effect:
            for iA, pA in enumerate(updated_posecodes):
                for iB, pB in enumerate(updated_posecodes[iA+1:]):
                    for iC, pC in enumerate(updated_posecodes[iA+iB+2:]): # study each triplet (of distinct elements) only once
                        # ripple effect happens if:
                        # - pA & pB (resp. pA & pC or pB & pC) have one side & body
                        #   part in common (that can't be None) - ie. there must be
                        #   exactly 3 different body parts at stake
                        # - pA, pB and pC have the same, or opposite interpretations
                        #   (eg. "below"/"above" is OK, but "below"/"behind" is not)
                        s = set([tuple(pA[:2]), tuple(pA[3:]),
                                tuple(pB[:2]), tuple(pB[3:]),
                                tuple(pC[:2]), tuple(pC[3:])])
                        if len(s) == 3 and tuple([None, None]) not in s and \
                            same_posecode_family(pA, pB) and same_posecode_family(pB, pC):
                            transrel_rer_removed +=1 # one posecode will be removed
                            # keep pA as is
                            # convert pB such that the interpretation is the same as pA
                            pB_prime = pB if pB[2] == pA[2] else reverse_joint_order(pB)
                            if pA[:2] == pB_prime[3:]:
                                # then pB_prime[:2] < pA[:2] = pB_prime[3:] < pA[3:]
                                updated_posecodes.remove(pC)
                                if extra_verbose: print("Removed (ripple effect):", pC)
                            else:
                                # convert pC such that the interpretation is the same as pA
                                pC_prime = pC if pC[2] == pA[2] else reverse_joint_order(pC)
                                if pB_prime[3:] == pC_prime[:2]:
                                    # then pA[3:] == pC_prime[3:], which means that
                                    # pB_prime[:2] = pA[:2] < pB_prime[3:] = pC_prime[:2] < pA[3:] = pC_prime[3:]
                                    updated_posecodes.remove(pA)
                                    if extra_verbose: print("Removed (ripple effect):", pA)
                                else:
                                    # then pA[:2] == pC_prime[:2], which means that
                                    # pB_prime[:2] = pA[:2] < pA[3:] = pC_prime[:2] < pB_prime[3:] = pC_prime[3:]
                                    updated_posecodes.remove(pB)
                                    if extra_verbose: print("Removed (ripple effect):", pB)
                        # Example:
                        # "the left hand is above the neck, the right hand is
                        # below the neck, the left hand is above the right
                        # hand", ie. R hand < neck < L hand ==> remove the R/L
                        # hand posecode


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 1) Entity-based aggregations
        if not simplified_captions:
            for iA, pA in enumerate(updated_posecodes):
                for pB in copy.deepcopy(updated_posecodes[iA+1:]): # study each pair (of distinct elements) only once       
                    # At least one body part (the first, the second or both),
                    # for both posecodes, need to belong (together) to a larger
                    # body part. Aggregate if:
                    # - the two posecodes have the same interpretation
                    # - either:
                    #   * the two first body parts belong (together) to a larger
                    #     body part (ie. same side for the two first body parts) ;
                    #     and the two second body parts are the same
                    #   * vice-versa, for the second body parts and the first body parts
                    #   * the two first body parts belong (together) to a larger
                    #     body part (ie. same side for the two first body parts) ;
                    #     and the two second body parts belong (together) to a larger
                    #     body part (ie. same side for the two second body parts)
                    if pA[0] == pB[0] and pA[2:4] == pB[2:4] \
                        and random.random() < PROP_AGGREGATION_HAPPENS:
                        body_part_1 = ENTITY_AGGREGATION.get((pA[1], pB[1]), False)
                        body_part_2 = ENTITY_AGGREGATION.get((pA[4], pB[4]), False)
                        aggregation_happened = False
                        # non-systematic and non-exclusive aggregations
                        if body_part_1 and (pA[4] == pB[4] or body_part_2):
                            updated_posecodes[iA][1] = body_part_1
                            aggregation_happened = True
                        if body_part_2 and (pA[1] == pB[1] or body_part_1):
                            updated_posecodes[iA][4] = body_part_2
                            aggregation_happened = True
                        # remove the second posecode only if some aggregation happened
                        if aggregation_happened:
                            updated_posecodes[iA][5].extend(pB[5])
                            updated_posecodes.remove(pB)
                    # Examples:
                    # a) "the left hand  is below the right hand"
                    #     +
                    #    "the left elbow is below the right hand"
                    #
                    # ==>"the left arm   is below the right hand"
                    #
                    # b) "the left hand  is below the right hand"
                    #     +
                    #    "the left elbow is below the right elbow"
                    #
                    # ==>"the left arm   is below the right arm"
                    #
                    # c) [CASE IN WHICH AGGREGATION DOES NOT HAPPEN, SO NO POSECODE SHOULD BE REMOVED]
                    #    "the right knee is bent, the right elbow is bent"


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 2) Symmetry-based aggregations
        if not simplified_captions:
            for iA, pA in enumerate(updated_posecodes):
                for pB in copy.deepcopy(updated_posecodes[iA+1:]): # study each pair (of distinct elements) only once
                    # aggregate if the two posecodes:
                    # - have the same interpretation
                    # - have the same second body part (side isn't important)
                    # - have the same first body part
                    # - have not the same first side
                    if pA[1:3] == pB[1:3] and pA[4] == pB[4] \
                        and random.random() < PROP_AGGREGATION_HAPPENS:
                        # remove side, and indicate to put the verb plural
                        updated_posecodes[iA][0] = PLURAL_KEY
                        updated_posecodes[iA][1] = pluralize(pA[1])
                        if updated_posecodes[iA][3] != pB[3]:
                            # the second body part is studied for both sides,
                            # so pluralize the second body part
                            # (if the body part doesn't have a side (ie. its
                            # side is set to None), it is necessarily None for
                            # both posecodes (since the second body part needs
                            # to be the same for both posecodes), and so the
                            # program doesn't end up here. Hence, no need to
                            # treat this case here.)
                            updated_posecodes[iA][3] = PLURAL_KEY
                            updated_posecodes[iA][4] = pluralize(pA[4])
                        updated_posecodes[iA][5].extend(pB[5])
                        updated_posecodes.remove(pB)


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 3) Discard posecodes based on "common sense" relation rules between posecodes (ripple effect #2)
        # Entity-based and symmetry-based aggregations actually defuse some of
        # the relation rules, hence the reason why these steps must happen
        # before discarding any posecode based on the relation rules.
        # These rules are "common sense" to the extent that they were found to
        # apply to a large majority of poses. They were automatically detected
        # based on statistics and manually cleaned.
        if apply_stat_ripple_effect:
            # remove posecodes based on ripple effect rules if:
            # - the pose satisfies the condition posecodes A & B
            #   (look at raw posecodes, before any aggregation to tell)
            # - the pose satisfies the resulting posecode C, and posecode C
            #   is still available (as a raw posecode) after entity-based &
            #   symmetry-based aggregation, and after potential application of
            #   other ripple effect rules (look at updated_posecodes to tell)
            # Note: no B posecode in bi-relations A ==> C ("B" is None)
            for rer in stat_rer:
                if rer[0] in posecodes[p] and \
                    (rer[1] is None or rer[1] in posecodes[p]) and \
                    rer[2] in updated_posecodes:
                    if extra_verbose:
                        print(f"Applied ripple effect rule {quick_posecode_display(rer[0])} + {quick_posecode_display(rer[1])} ==> {quick_posecode_display(rer[2])}.")
                    updated_posecodes.remove(rer[2])
                    stat_rer_removed += 1


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 4) Random side-reversing 
        # It is a form of polishing step that must happen before other kinds of
        # aggregations such as interpretation-based aggregations and
        # focus-body-part-based aggregations (otherwise, they will be a bias
        # toward the left side, which is always defined as the first body part
        # for code simplicity and consistency).
        for i_pc, pc in enumerate(updated_posecodes):                
            # Swap the first & second joints when they only differ about their
            # side; and adapt the interpretation.
            if pc[1] == pc[4] and pc[0] != pc[3] and random.random() < 0.5:
                pc[:2], pc[3:5] = pc[3:5], pc[:2]
                pc[2] = OPPOSITE_CORRESP_ID[pc[2]]
                updated_posecodes[i_pc] = pc
            # Randomly process two same body parts as a single body part if
            # allowed by the corresponding posecode interpretation (ie. randomly
            # choose between 1-component and 2-component template sentences, eg.
            # "L hand close to R hand" ==> "the hands are close to each other")
            if pc[2] in OK_FOR_1CMPNT_OR_2CMPNTS_IDS and pc[1] == pc[4] and random.random() < 0.5:
                # remove side, indicate to put the verb plural, and remove the
                # second component
                updated_posecodes[i_pc] = [PLURAL_KEY, pluralize(pc[1]), pc[2], None, None, pc[5]]


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 5) Interpretation-based & focus-body-part-based aggregations
        if not simplified_captions:
            updated_posecodes = aggreg_fbp_intptt_based(updated_posecodes, extra_verbose=extra_verbose)


        # eventually, apply all changes
        posecodes[p] = updated_posecodes
        if extra_verbose: 
            print("Final posecodes:")
            print(updated_posecodes)

    # Display stats on ripple effect rules
    print("Posecodes removed by ripple effect rules based on statistics: ", stat_rer_removed)
    print("Posecodes removed by ripple effect rules based on transitive relations:", transrel_rer_removed)

    return posecodes


def aggreg_fbp_intptt_based(posecodes_1p, extra_verbose=False):
    """
    posecodes_1p: list of posecodes (structures of size 5) for a single pose

    NOTE: interpretation-based aggregations and joint-based aggregations are not
    independent, and could be applied on similar set of posecodes. Hence, one
    cannot happen before the other. They need to be processed together
    simultaneously.
    NOTE: interpretation-based and joint-based aggregations are the mirror of
    each other; interpretation-based aggregation can be schematized as follow:
    x ~ y & x ~ z ==> x ~ (y+z)
    while joint-based aggregation can be schematized as follow:
    y ~ x & z ~ x ==> (y+z) ~ x
    where "a ~ b" symbolises the relation (which depends on b) between body
    side & part 1 (a) and body side & part 2 (b)
    """

    # list eligible interpretation-based and focus-body-part-based
    # aggregations by listing the different sets of aggregable posecodes
    # (identified by their index in the posecode list) for each
    intptt_a = {}
    fbp_a = {}
    for p_ind, p in enumerate(posecodes_1p):
        # interpretation-based aggregations require the second body part to
        # be the same (a bit like entity-based aggregations between elements
        # that do not form together a larger standard entity)
        intptt_a[tuple(p[2:5])] = intptt_a.get(tuple(p[2:5]), []) + [p_ind]
        fbp_a[tuple(p[:2])] = fbp_a.get(tuple(p[:2]), []) + [p_ind]

    # choose which aggregations will be performed among the possible ones
    # to this end, shuffle the order in which the aggregations will be considered;
    # there must be at least 2 posecodes to perform an aggregation
    possible_aggregs = [('intptt', k) for k,v in intptt_a.items() if len(v)>1] + \
                        [('fbp', k) for k,v in fbp_a.items() if len(v)>1]
    random.shuffle(possible_aggregs) # potential aggregations will be studied in random order, independently of their kind
    aggregs_to_perform = [] # list of the aggregations to perform later (either intptt-based or fbp-based)
    unavailable_p_inds = set() # indices of the posecodes that will be aggregated
    for agg in possible_aggregs:
        # get the list of posecodes ids that would be involved in this aggregation
        p_inds = intptt_a[agg[1]] if agg[0] == "intptt" else fbp_a[agg[1]]
        # check that all or a part of them are still available for aggregation
        p_inds = list(set(p_inds).difference(unavailable_p_inds))
        if len(p_inds) > 1: # there must be at least 2 (unused, hence available) posecodes to perform an aggregation
            # update list of posecode indices to aggregate
            random.shuffle(p_inds) # shuffle to later aggregate these posecodes in random order
            if agg[0] == "intptt":
                intptt_a[agg[1]] = p_inds
            elif agg[0] == "fbp":
                fbp_a[agg[1]] = p_inds
            # grant aggregation (to perform later)
            unavailable_p_inds.update(p_inds)
            aggregs_to_perform.append(agg)
    
    # perform the elected aggregations
    if extra_verbose: print("Aggregations to perform:", aggregs_to_perform)
    updated_posecodes = []
    for agg in aggregs_to_perform:
        if random.random() < PROP_AGGREGATION_HAPPENS: 
            if agg[0] == "intptt":
                # perform the interpretation-based aggregation
                # agg[1]: (size 3) interpretation id, side2, body_part2
                # new_posecode = [MULTIPLE_SUBJECTS_KEY, [posecodes_1p[p_ind][:2] for p_ind in intptt_a[agg[1]]]] + list(agg[1])
                new_posecode = [MULTIPLE_SUBJECTS_KEY,
                                [posecodes_1p[p_ind][:2] for p_ind in intptt_a[agg[1]]]] + \
                               list(agg[1]) + \
                               [[p_add for p_ind in intptt_a[agg[1]] for p_add in posecodes_1p[p_ind][5]]]
            elif agg[0] == "fbp":
                # perform the focus-body-part-based aggregation
                # agg[1]: (size 2) side1, body_part1
                # new_posecode = [JOINT_BASED_AGGREG_KEY, list(agg[1]),
                #                 [posecodes_1p[p_ind][2] for p_ind in fbp_a[agg[1]]],
                #                 [posecodes_1p[p_ind][3:] for p_ind in fbp_a[agg[1]]]]
                new_posecode = [JOINT_BASED_AGGREG_KEY, list(agg[1]),
                                [posecodes_1p[p_ind][2] for p_ind in fbp_a[agg[1]]],
                                [posecodes_1p[p_ind][3:5] for p_ind in fbp_a[agg[1]]],
                                [p_add for p_ind in fbp_a[agg[1]] for p_add in posecodes_1p[p_ind][5]]
                               ]
                # if performing interpretation-fusion, it should happen here
                # ie. ['<joint_based_aggreg>', ['right', 'arm'], [16, 9, 15], [['left', 'arm'], ['left', 'arm'], ['left', 'arm']]], 
                # which leads to "the right arm is behind the left arm, spread far apart from the left arm, above the left arm"
                # whould become something like "the right arm is spread far apart from the left arm, behind and above it"
                # CONDITION: the second body part is not None, and is the same for at least 2 interpretations
                # CAUTION: one should avoid mixing "it" words refering to BP2 with "it" words refering to BP1...
            updated_posecodes.append(new_posecode)
    if extra_verbose:
        print("Posecodes from interpretation/joint-based aggregations:")
        for p in updated_posecodes:
            print(p)

    # don't forget to add all the posecodes that were not subject to these kinds
    # of aggregations
    updated_posecodes.extend([p for p_ind, p in enumerate(posecodes_1p) if p_ind not in unavailable_p_inds])

    return updated_posecodes



## AGGREGATE MOTIONCODES
################################################################################

def aggregate_motioncodes(posecode_info, motioncodes, time_bin_info, simplified_captions=False,
                        apply_transrel_ripple_effect=True, apply_stat_ripple_effect=True, agg_deactivated=False,
                        extra_verbose=False):
    # augment ripple effect rules to have rules for the R side as well
    # (rules registered in the captioning_data were found to apply for L & R,
    # but were registered for the L side only for declaration simplicity)
    stat_rer = STAT_BASED_RIPPLE_EFFECT_RULES + [[get_symmetric_posecode(pc) for pc in l] for l in
                                                 STAT_BASED_RIPPLE_EFFECT_RULES]
    # format ripple effect rules for later processing
    stat_rer = [[get_posecode_from_name(pc) if pc != "---" else None for pc in l] for l in stat_rer]
    # get stats over posecode discarding based on application of ripple effect rules (rer)
    stat_rer_removed = 0  # rules based on statistically frequent pairs and triplets of posecodes
    transrel_rer_removed = 0  # rules based on transitive relations between body parts




    Motion_Bins = [[] for _ in range(time_bin_info['nb_binds'])]

    nb_motions = len(motioncodes)
    for m in range(nb_motions):
        bin_number = motioncodes[m][2]['start'] // time_bin_info['bin_size']
        Motion_Bins[bin_number].append(motioncodes[m])




    # treat each time windows one by one
    if True:
        for time_window_index in range(len(Motion_Bins)):
            current_window = Motion_Bins[time_window_index]
            nb_motions = len(current_window)
            updated_current_window = copy.deepcopy(current_window)

            # for m in range(nb_motions):
            if True:
                if extra_verbose:
                    print(f'Bin  {time_window_index}')
                    # print(f"\n**Motion {m}")
                    print("Initial motioncodes at window:")
                    print(current_window)

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # TODO: ----------- Subject Detection: Done! -----------
                # TODO: this is where we should appply the subject joints vs objects.
                # TODO: e.g. whether left elbow goes to teh right one, the right
                #  elbow goes to the left one, or both have the samee contribution.
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # 4) XXX*Random*XXX side-reversing
                # It is a form of polishing step that must happen before other kinds of
                # aggregations such as interpretation-based aggregations and
                # focus-body-part-based aggregations (otherwise, they will be a bias
                # toward the left side, which is always defined as the first body part
                # for code simplicity and consistency).

                # It is important to find the correct active joint as subject
                # this should be happened prior to any other aggregation, e.g.
                # left elbow closing to the right knee +
                # right elbow closing to the right knee =
                # the elbows closing to the right knee might inferred at the symmetric step
                # This might be incorrect somehow if on of elbows is not the
                # active joint.
                # Therefore, we should apply this at the very beginning where
                # each motioncode has a list of one item as posecode entery,
                # except super-motion codes which are not defined yet.

                # However, note that we do not swap the involved joints if it is a body-focused
                # motioncode such as Proximity and Spatial Relation on some joints:
                # e.g. left-kne "to the right of" right knee vs. left hand from below to above none neck
                # where the 1st one could swap based on the body focus while the second one is
                # a body part focused on the left hand. Therefore, we need to check whether the
                # motioncode is a body-focused


                for i_mc, mc in enumerate(updated_current_window):

                    # Todo 0: body-focus query check - ----Fixed
                    # First of all, we need to check if this query has a body-part
                    # focus or not, this is because we added more complicated rules
                    # that are consist of body-part focused or not.
                    try:
                        if mc[2]['mc_info']['focus_body_part'] is not None:
                            continue
                    except:
                        print()

                    # Todo 1.
                    # Swap the first & second joints when they only differ about their
                    # side; and adapt the interpretation.
                    # ---> we add the "detected activity" equality of two joints as an extra condition.
                    # Otherwise, the subject will be set to the most active joints.

                    # ToDo: we need to decide whether this one is a symmetric or non-symmetric motioncode

                    # 1. Detect joint_set
                    # -------------------
                    st = mc[2]['start']
                    et = mc[2]['end']
                    reqired_pose = mc[2]['posecode'][0]  # since it is only one piece
                    p_kind = reqired_pose[0]

                    if p_kind not in SUBJECT_OBJECT_REQUIRED_KINDS:  # Skip p_kinds without object joint e.g. angular
                        continue


                    #TODO: I do not understand why is this here?!?!?!?!?!
                    # because all the two-body-part required rules were in that too?
                    # I think I know, it should be somewhere we want to combine 1st
                    # and 2nd body parts together and put a Plural_Key there.
                    # if mc[2]['spatial'] not in OK_FOR_1CMPNT_OR_2CMPNTS_IDS_MOTIONCODES:
                    #     continue  # todo: this should go somewhere else
                    # Commented out due to the reason it only required when we want to
                    # combine two symmetric/non-symmetric 1st and 2nd body part together when
                    # they have almost equal contribution in the displacement motion.





                    js = reqired_pose[1]
                    involved_js = posecode_info['p_queries'][p_kind]['joint_ids'][js]

                    # 2. Evaluate motions signal of each joint
                    # ----------------------------------------
                    # Now we do the signal eval over each of those joints
                    sum_movement = [0, 0]
                    for i_ij, involved_joint in enumerate(involved_js):

                        signal_x = posecode_info['p_interpretations']['position_x'][st:et, involved_joint].cpu().numpy()
                        signal_y = posecode_info['p_interpretations']['position_y'][st:et, involved_joint].cpu().numpy()
                        signal_z = posecode_info['p_interpretations']['position_z'][st:et, involved_joint].cpu().numpy()
                        # --------------------------------------------------------------------------------------------
                        delta_signal_x = [0] + [signal_x[i + 1] - signal_x[i] for i in
                                                range(len(signal_x) - 1)]
                        delta_signal_y = [0] + [signal_y[i + 1] - signal_y[i] for i in
                                                range(len(signal_y) - 1)]
                        delta_signal_z = [0] + [signal_z[i + 1] - signal_z[i] for i in
                                                range(len(signal_z) - 1)]
                        # -------------------------------------------------------------------------------------------
                        motions_x = single_path_finder(delta_signal_x)
                        motions_y = single_path_finder(delta_signal_y)
                        motions_z = single_path_finder(delta_signal_z)
                        M3D = (motions_x + motions_y + motions_z)
                        for detected_motion in M3D:
                            sum_movement[i_ij] += detected_motion['intensity'] ** 2

                    # 3. set values for active joints
                    # -------------------
                    sum_movement[0] = math.sqrt(sum_movement[0])
                    sum_movement[1] = math.sqrt(sum_movement[1])
                    total_movement = sum(sum_movement)
                    swap = sum_movement[1] > sum_movement[0]
                    equal = sum_movement[0] > 0.4 * total_movement and sum_movement[1] > 0.4 * total_movement

                    # Ablation-study on Subject-Detection
                    # equal = False
                    # swap = random.choice([True, False])



                    if equal:  # we check equality (thresholded-equality) first,
                        # otherwise most of the time there would be at least very
                        # small difference between two joints

                        # Randomly process two same body parts as a single body part if
                        # allowed by the corresponding posecode interpretation (ie. randomly
                        # choose between 1-component and 2-component template sentences, eg.
                        # "L hand close to R hand" ==> "the hands are close to each other")
                        # Todo: it seems we do not need OK_FOR_1CMPNT_OR_2CMPNTS_IDS_MOTIONCODE
                        #  since they should be able to convert into 1Component type.
                        #  We keep it for now just in case
                        #  We should put all possible spatial interpretations into this at the
                        #  beginning and it is important since we would need it
                        #  e.g. L hand from behind the head to the front of it
                        #  which is a relative axis posecode

                        # Check if it is possible to use a 1CMPNT
                        if mc[2]['spatial'] in OK_FOR_1CMPNT_OR_2CMPNTS_IDS_MOTIONCODES:
                            # 1. when they are symmetric
                            if mc[1] == mc[4] and random.random() < 0.95:
                                # remove side, indicate to put the verb plural, and remove the
                                # second component
                                mc = [PLURAL_KEY, pluralize(mc[1]), mc[2], None, None]
                            # 2. when they are non-symmetric
                            else:
                                # todo: this should be interpretation-based aggregation ????????
                                # we might also consider a bit of randomization for the order
                                # of appearance of each joint
                                empty = ''
                                mc = [PLURAL_KEY,
                                      f'{empty if mc[0] == None else mc[0]} {mc[1]} and {empty if mc[3] == None else mc[3]} {mc[4]}',
                                      mc[2], None, None]
                        elif random.random() < 0.5:  # when they are not allowed to be combined
                            mc[:2], mc[3:5] = mc[3:5], mc[:2]
                            if mc[2]['spatial'] in OPPOSITE_CORRESP_ID_MOTIONCODES:
                                mc[2]['spatial'] = OPPOSITE_CORRESP_ID_MOTIONCODES[mc[2]['spatial']]
                    elif swap:
                        mc[:2], mc[3:5] = mc[3:5], mc[:2]
                        if mc[2]['spatial'] in OPPOSITE_CORRESP_ID_MOTIONCODES:
                            mc[2]['spatial'] = OPPOSITE_CORRESP_ID_MOTIONCODES[mc[2]['spatial']]

                    # Apply
                    updated_current_window[i_mc] = mc
                    #     Done!

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # 0) Remove redundant information (ripple effect #1)
                # If we have 3 posecodes telling that a < b, b < c and a < c (with 'a',
                # 'b', and 'c' being arbitrary body parts, and '<' representing an order
                # relation such as 'below'), then keep only the posecodes telling that a
                # < b and b < c, as it is enough to infer the global relation a < b < c.
                if False: # We skip the ripple efect rules for now since we haven't done the stat. analysis yet.
                    if apply_transrel_ripple_effect:
                        for iA, pA in enumerate(updated_posecodes):
                            for iB, pB in enumerate(updated_posecodes[iA + 1:]):
                                for iC, pC in enumerate(
                                        updated_posecodes[iA + iB + 2:]):  # study each triplet (of distinct elements) only once
                                    # ripple effect happens if:
                                    # - pA & pB (resp. pA & pC or pB & pC) have one side & body
                                    #   part in common (that can't be None) - ie. there must be
                                    #   exactly 3 different body parts at stake
                                    # - pA, pB and pC have the same, or opposite interpretations
                                    #   (eg. "below"/"above" is OK, but "below"/"behind" is not)
                                    s = set([tuple(pA[:2]), tuple(pA[3:]),
                                             tuple(pB[:2]), tuple(pB[3:]),
                                             tuple(pC[:2]), tuple(pC[3:])])
                                    if len(s) == 3 and tuple([None, None]) not in s and \
                                            same_posecode_family(pA, pB) and same_posecode_family(pB, pC):
                                        transrel_rer_removed += 1  # one posecode will be removed
                                        # keep pA as is
                                        # convert pB such that the interpretation is the same as pA
                                        pB_prime = pB if pB[2] == pA[2] else reverse_joint_order(pB)
                                        if pA[:2] == pB_prime[3:]:
                                            # then pB_prime[:2] < pA[:2] = pB_prime[3:] < pA[3:]
                                            updated_posecodes.remove(pC)
                                            if extra_verbose: print("Removed (ripple effect):", pC)
                                        else:
                                            # convert pC such that the interpretation is the same as pA
                                            pC_prime = pC if pC[2] == pA[2] else reverse_joint_order(pC)
                                            if pB_prime[3:] == pC_prime[:2]:
                                                # then pA[3:] == pC_prime[3:], which means that
                                                # pB_prime[:2] = pA[:2] < pB_prime[3:] = pC_prime[:2] < pA[3:] = pC_prime[3:]
                                                updated_posecodes.remove(pA)
                                                if extra_verbose: print("Removed (ripple effect):", pA)
                                            else:
                                                # then pA[:2] == pC_prime[:2], which means that
                                                # pB_prime[:2] = pA[:2] < pA[3:] = pC_prime[:2] < pB_prime[3:] = pC_prime[3:]
                                                updated_posecodes.remove(pB)
                                                if extra_verbose: print("Removed (ripple effect):", pB)
                                    # Example:
                                    # "the left hand is above the neck, the right hand is
                                    # below the neck, the left hand is above the right
                                    # hand", ie. R hand < neck < L hand ==> remove the R/L
                                    # hand posecode

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # 1) Entity-based aggregations
                if not simplified_captions and not agg_deactivated:
                    for iA, mA in enumerate(updated_current_window):
                        for mB in copy.deepcopy(
                                updated_current_window[iA + 1:]):  # study each pair (of distinct elements) only once
                            # At least one body part (the first, the second or both),
                            # for both motioncodes, need to belong (together) to a larger
                            # body part. Aggregate if:
                            # - the two motioncodes have the same interpretation
                            # - either:
                            #   * the two first body parts belong (together) to a larger
                            #     body part (ie. same side for the two first body parts) ;
                            #     and the two second body parts are the same
                            #   * vice-versa, for the second body parts and the first body parts
                            #   * the two first body parts belong (together) to a larger
                            #     body part (ie. same side for the two first body parts) ;
                            #     and the two second body parts belong (together) to a larger
                            #     body part (ie. same side for the two second body parts)
                            if mA[0] == mB[0] and ( mA[3:4] == mB[3:4] and  mA[2]['spatial'] == mB[2]['spatial'] )\
                                    and random.random() < PROP_AGGREGATION_HAPPENS:
                                body_part_1 = ENTITY_AGGREGATION.get((mA[1], mB[1]), False)
                                body_part_2 = ENTITY_AGGREGATION.get((mA[4], mB[4]), False)
                                aggregation_happened = False
                                # non-systematic and non-exclusive aggregations
                                if body_part_1 and (mA[4] == mB[4] or body_part_2):
                                    updated_current_window[iA][1] = body_part_1
                                    aggregation_happened = True
                                if body_part_2 and (mA[1] == mB[1] or body_part_1):
                                    updated_current_window[iA][4] = body_part_2
                                    aggregation_happened = True
                                # remove the second posecode only if some aggregation happened
                                if aggregation_happened:
                                    updated_current_window[iA][2]['posecode'].extend(mB[2]['posecode'])
                                    updated_current_window.remove(mB)
                            # Examples:
                            # a) "the left hand  is below the right hand"
                            #     +
                            #    "the left elbow is below the right hand"
                            #
                            # ==>"the left arm   is below the right hand"
                            #
                            # b) "the left hand  is below the right hand"
                            #     +
                            #    "the left elbow is below the right elbow"
                            #
                            # ==>"the left arm   is below the right arm"
                            #
                            # c) [CASE IN WHICH AGGREGATION DOES NOT HAPPEN, SO NO POSECODE SHOULD BE REMOVED]
                            #    "the right knee is bent, the right elbow is bent"

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # 2) Symmetry-based aggregations
                if not agg_deactivated:
                    if not simplified_captions:
                        for iA, mA in enumerate(updated_current_window):
                            for mB in copy.deepcopy(updated_current_window[iA + 1:]):  # study each pair (of distinct elements) only once
                                # aggregate if the two motioncodes:
                                # - have the same interpretation
                                #   - it immediately rises the question how we would like
                                #     to assume two interpretations are the same w.r.t.
                                #     the action i.e. bending/extension, intensity, and velocity
                                #           ** We may consider something like OPPOSITE_CORRESP
                                # - have the same second body part (side isn't important)
                                # - have the same first body part
                                # - have not the same first side
                                if mA[1:2] == mB[1:2] and mA[2]['spatial'] == mB[2]['spatial'] and mA[4] == mB[4] \
                                        and random.random() < PROP_AGGREGATION_HAPPENS:
                                    # remove side, and indicate to put the verb plural
                                    updated_current_window[iA][0] = PLURAL_KEY
                                    updated_current_window[iA][1] = pluralize(mA[1])
                                    if updated_current_window[iA][3] != mB[3]:
                                        # the second body part of (side, joint) is studied
                                        # for both sides, so pluralize the second body part
                                        # (if the body part doesn't have a side (ie. its
                                        # side is set to None), it is necessarily None for
                                        # both posecodes (since the second body part needs
                                        # to be the same for both posecodes), and so the
                                        # program doesn't end up here. Hence, no need to
                                        # treat this case here.)
                                        updated_current_window[iA][3] = PLURAL_KEY
                                        updated_current_window[iA][4] = pluralize(mA[4])
                                    updated_current_window[iA][2]['posecode'].extend(mB[2]['posecode'])
                                    updated_current_window.remove(mB)

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # 3) Discard posecodes based on "common sense" relation rules between posecodes (ripple effect #2)
                # Entity-based and symmetry-based aggregations actually defuse some of
                # the relation rules, hence the reason why these steps must happen
                # before discarding any posecode based on the relation rules.
                # These rules are "common sense" to the extent that they were found to
                # apply to a large majority of poses. They were automatically detected
                # based on statistics and manually cleaned.
                if False:
                    if apply_stat_ripple_effect:
                        # remove posecodes based on ripple effect rules if:
                        # - the pose satisfies the condition posecodes A & B
                        #   (look at raw posecodes, before any aggregation to tell)
                        # - the pose satisfies the resulting posecode C, and posecode C
                        #   is still available (as a raw posecode) after entity-based &
                        #   symmetry-based aggregation, and after potential application of
                        #   other ripple effect rules (look at updated_posecodes to tell)
                        # Note: no B posecode in bi-relations A ==> C ("B" is None)
                        for rer in stat_rer:
                            if rer[0] in posecodes[p] and \
                                    (rer[1] is None or rer[1] in posecodes[p]) and \
                                    rer[2] in updated_posecodes:
                                if extra_verbose:
                                    print(
                                        f"Applied ripple effect rule {quick_posecode_display(rer[0])} + {quick_posecode_display(rer[1])} ==> {quick_posecode_display(rer[2])}.")
                                updated_posecodes.remove(rer[2])
                                stat_rer_removed += 1



                Motion_Bins[time_window_index] = copy.deepcopy(updated_current_window)


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # We move this step out of the for loop
    # 5) Interpretation-based & focus-body-part-based aggregations
    if not agg_deactivated:
        if not simplified_captions:
            updated_motioncodes = aggreg_fbp_intptt_time_based(Motion_Bins, time_bin_info['max_range_bins'], extra_verbose=extra_verbose)
    else:
        # Preparing it such that we could use them in further steps
        updated_motioncodes = [[] for _ in range(len(Motion_Bins))]
        motioncodes_1p = Motion_Bins
        for i in range(len(motioncodes_1p)):
            for j in range(len(motioncodes_1p[i])):
                # if tuple([i, j, motioncodes_1p[i][j][2]['spatial'], 'point2M1']) not in covered_motioncodes:
                    # if tuple([i, j, motioncodes_1p[i][j][2]['spatial'], 'point2M1']) not in unavailable_motion_inds:
                mc = motioncodes_1p[i][j]
                if not isinstance(mc[2], list):  # if not a result from the Agg. step
                    mc[2]['bin_diff'] = 0
                    mc = ['<SINGLE>', [[mc[0], mc[1]]], [mc[2]], [[mc[3], mc[4]]], set(mc[2]['posecode'])]
                updated_motioncodes[i].append(mc)


    # eventually, apply all changes
    Motion_Bins = updated_motioncodes
    if extra_verbose:
        print("Final Motion-Codes:")
        print(updated_motioncodes)

    # treat time windows through time
    # raise NotImplementedError


    # Display stats on ripple effect rules
    print("Posecodes removed by ripple effect rules based on statistics: ", stat_rer_removed)
    print("Posecodes removed by ripple effect rules based on transitive relations:", transrel_rer_removed)

    return Motion_Bins


def aggreg_fbp_intptt_time_based(motioncodes_1p, max_range_bins, extra_verbose=False):
    """
    posecodes_1p: list of posecodes (structures of size 5) for a single pose

    NOTE: interpretation-based aggregations and joint-based aggregations are not
    independent, and could be applied on similar set of posecodes. Hence, one
    cannot happen before the other. They need to be processed together
    simultaneously.
    NOTE: interpretation-based and joint-based aggregations are the mirror of
    each other; interpretation-based aggregation can be schematized as follow:
    x ~ y & x ~ z ==> x ~ (y+z)
    while joint-based aggregation can be schematized as follow:
    y ~ x & z ~ x ==> (y+z) ~ x
    where "a ~ b" symbolises the relation (which depends on b) between body
    side & part 1 (a) and body side & part 2 (b)
    """

    # list eligible interpretation-based and focus-body-part-based
    # aggregations by listing the different sets of aggregable posecodes
    # (identified by their index in the posecode list) for each

    # ------------------------------------------------------------
    # 1. Inside time windows Agg. over interpretations and joints

    all_intptt_a = [{} for _ in range(len(motioncodes_1p))]
    all_fbp_a = [{} for _ in range(len(motioncodes_1p))]

    for bin_ind, bin in enumerate(motioncodes_1p):
        intptt_a = {}
        fbp_a = {}
        for m_ind, motion in enumerate(bin):
            # interpretation-based aggregations require the second body part to
            # be the same (a bit like entity-based aggregations between elements
            # that do not form together a larger standard entity)

            # intptt_a[tuple(p[2:])] = intptt_a.get(tuple(p[2:]), []) + [p_ind]
            # fbp_a[tuple(p[:2])] = fbp_a.get(tuple(p[:2]), []) + [p_ind]


            # since we aggregate with the maximum possible joint sets of a specific interpretation
            # it prevents other possibilities to get the chance of being aggregated through time
            # So agg "inside" get the only eligible agg among [inside, over-bins, inside + over-bins]
            # We should produce all possible combination of each aggregation
            #  For example: Bin_1 [('RElbow', bending), ('LElbow', bending), ('RKnee', bending)] and
            #               Bin_1 [('RElbow', extending), ('LElbow', extending)]
            # Current Agg. over interpretation is always considers non-aggregated vs. Agg. with the
            # largest possible joint set aggregation for a specific interpretation inside a time window.
            # Therefore, we have to produce all combinations
            # intptt_a[tuple([motion[2]['spatial']]+motion[3:])] = intptt_a.get(tuple([motion[2]['spatial']]+motion[3:]), []) + [(bin_ind, m_ind, motion[2]['spatial'])]
            intptt_a[len(intptt_a)] = {'key': tuple([motion[2]['spatial']]+motion[3:]), 'list': [(bin_ind, m_ind, motion[2]['spatial'], 'point2M1')] }
            key_2_search = tuple([motion[2]['spatial']]+motion[3:])
            for i in range(len(intptt_a)-1): # -1 to avoid agg. with itself
                if intptt_a[i]['key'] == key_2_search:
                    intptt_a[len(intptt_a)] = {'key': key_2_search, 'list': intptt_a[i]['list'] + [(bin_ind, m_ind, motion[2]['spatial'], 'point2M1')] }

            # fbp_a[tuple(motion[:2])] = fbp_a.get(tuple(motion[:2]), []) + [(bin_ind, m_ind, motion[2]['spatial'])]
            fbp_a[len(fbp_a)] = { 'key': tuple(motion[:2]), 'list': [(bin_ind, m_ind, motion[2]['spatial'], 'point2M1')] }
            key_2_search =  tuple(motion[:2])
            for i in range(len(fbp_a) - 1):  # -1 to avoid agg. with itself
                if fbp_a[i]['key'] == key_2_search:
                    fbp_a[len(fbp_a)] = {'key': key_2_search,
                                               'list': fbp_a[i]['list'] + [(bin_ind, m_ind, motion[2]['spatial'], 'point2M1')]}

        all_intptt_a[bin_ind] = intptt_a
        all_fbp_a[bin_ind] = fbp_a
    # ------------------------------------------------------------
    # 2. Through time windows Agg. over interpretations and joints

    backup_bins = copy.deepcopy(motioncodes_1p)
    time_a = [{} for _ in range(len(motioncodes_1p))]

    time_b = [{} for _ in range(len(motioncodes_1p))]
    for bin1_ind in range(len(motioncodes_1p)):
        # RND max_range here

            # todo: due to focus body and second side zero, we should agg based on the first side (None)

            # 1. Time aggregation over bins on interpretations aggregations
            for m1_ind in (all_intptt_a[bin1_ind]):
                # all_intptt_a[bin1_ind][motion1]
                # if (bin1_ind, m1_ind) in visited: continue
                # else: visited.append((bin1_ind, m1_ind))
                motion1 = all_intptt_a[bin1_ind][m1_ind]
                # time_a[bin1_ind][tuple(motion1[1:])] = time_a[bin1_ind].get(tuple(motion1[1:]), [{ 'bin_diff': 0, 'motion': motion1}])
                time_a[bin1_ind][len(time_a[bin1_ind])] = {'key': tuple(motion1['key'][1:]), 'list': [{ 'bin_diff': 0, 'motion': (bin1_ind, m1_ind, 'all_intptt_a')}]}
                specific_x = len(time_a[bin1_ind]) - 1  # we use this as the index of the latest extended list of motions
                key_2_search = tuple(motion1['key'][1:]) # infact joints
                # for bin2_ind in range(bin1_ind + 1, min(bin1_ind + max_range_bins, len(motioncodes_1p))):
                for bin2_ind in range(bin1_ind, min(bin1_ind + max_range_bins, len(motioncodes_1p))):
                    for m2_i in range(len(all_intptt_a[bin2_ind])):  # -1 to avoid agg. with itself
                        # However, there might be more than one motion with the current jointset, so we
                        # should iterate even over bin1 while skipping itslef
                        # This way we aggregate the "same time" motions
                        if bin2_ind==bin1_ind and m2_i==m1_ind:
                            continue    # TODO: not sure about starting from bin1_ind


                        # check if it is already aggregted inside the bin or it is a new joint set
                        # Another concern would be if js are equal but not covering new motion codes
                        # if bin1_ind==bin2_ind:
                        #     covered_m1 = all_intptt_a[bin1_ind][m1_ind]['list']
                        #     covered_m2 = all_intptt_a[bin2_ind][m2_i]['list']
                        #     agg_covered = list(set(covered_m1+covered_m2))
                        #     for jx in range(len(all_intptt_a[bin1_ind])):
                        #         covered_mx = all_intptt_a[bin1_ind][jx]['list']
                        #         all_elements_in_list2 = all(element in covered_mx for element in agg_covered)
                        #         if all_elements_in_list2:
                        #             break
                        #     if all_elements_in_list2:
                        #         continue
                        #     print()
                        #

                        # covered_m2 = all_intptt_a[bin2_ind][m2_i]['list']
                        # addition = list(set(covered_m2).difference(covered_m1))
                        # if len(addition)==0:
                        #     continue

                        if all_intptt_a[bin2_ind][m2_i]['key'][1:] == key_2_search:
                            # if motion1[1:] == motion2[1:]: # skip the interpretation
                            # We should check the joints consistency through all windows here
                            first_bin_aggs = all_intptt_a[bin1_ind][m1_ind]['list']
                            first_bin_joints = set([(motioncodes_1p[x[0]][x[1]][0], motioncodes_1p[x[0]][x[1]][1])
                                                    for x in first_bin_aggs] +
                                                   [(motioncodes_1p[x[0]][x[1]][3], motioncodes_1p[x[0]][x[1]][4])
                                                    for x in first_bin_aggs]
                                                    ) # sides and body part
                            second_bin_aggs = all_intptt_a[bin2_ind][m2_i]['list']
                            second_bin_joints = set([(motioncodes_1p[x[0]][x[1]][0], motioncodes_1p[x[0]][x[1]][1])
                                                    for x in second_bin_aggs] +
                                                    [(motioncodes_1p[x[0]][x[1]][3], motioncodes_1p[x[0]][x[1]][4])
                                                     for x in second_bin_aggs]
                                                    )
                            # Todo: maybe the second body part is not necessary?!?!?
                            # It is important e.g. closing to A or B, A should be = B
                            if first_bin_joints != second_bin_joints: continue

                            #TODO: IMPORTANT Here is where we should double-check if there is another motioncode
                            # in between this two and needed to be presented. XYZQ!@#$

                            time_a[bin1_ind][len(time_a[bin1_ind])] = {'key': key_2_search,
                                                                       'list': time_a[bin1_ind][specific_x]['list'] + [
                                                                           {'bin_diff': bin2_ind - bin1_ind,
                                                                            'motion': (bin2_ind, m2_i, 'all_intptt_a')}]}
                            specific_x = len(time_a[bin1_ind]) - 1 # Moving forward in time for that specific key2search
            # visited = [] # here we work with motion instead of it's index
            #2. Time aggregation over bins on focus body part aggregations
            for m1_ind in (all_fbp_a[bin1_ind]):
                # if all_fbp_a[bin1_ind][motion1] == None: continue
                # if (bin1_ind, motion1) in visited: continue
                # else: visited.append((bin1_ind, m1_ind))
                # if (bin2_ind, motion1) in visited: continue
                motion1 = all_fbp_a[bin1_ind][m1_ind]
                time_b[bin1_ind][len(time_b[bin1_ind])]= {'key': tuple(motion1['key']), 'list': [({'bin_diff': 0, 'motion': (bin1_ind, m1_ind, 'all_fbp_a')})]}
                specific_x = len(time_b[bin1_ind]) -1 # we use this as the index of the latest extended list of motions
                # for m2_ind, motion2 in enumerate(all_fbp_a[bin2_ind]):
                #     if all_fbp_a[bin2_ind][motion2] == None: continue
                #     if motion1 == motion2:  # skip the interpretation

                # if motion1 in all_fbp_a[bin2_ind]:
                key_2_search = tuple(motion1['key'])
                for bin2_ind in range(bin1_ind+1, min(bin1_ind + max_range_bins, len(motioncodes_1p))):
                # for bin2_ind in range(bin1_ind, min(bin1_ind + max_range_bins, len(motioncodes_1p))):
                    for m2_i in (all_fbp_a[bin2_ind]):

                        # if bin2_ind == bin1_ind and m2_i == m1_ind: continue  # TODO: not sure about starting from bin1_ind
                        # # but we want to consider also same time motions
                        # # Another concern would be if js are equal but not covering new motion codes
                        # covered_m1 = all_fbp_a[bin1_ind][m1_ind]['list']
                        # covered_m2 = all_fbp_a[bin2_ind][m2_i]['list']
                        # addition = list(set(covered_m2).difference(covered_m1))
                        # if len(addition) == 0:
                        #     continue
                        # commented since inner aggregation "sametime" is already happened

                        if all_fbp_a[bin2_ind][m2_i]['key'] == key_2_search:

                            # This one is always joint consistent since it is actually joint based

                            # time_b[bin1_ind][tuple(motion1)] += ([({ 'bin_diff': bin2_ind- bin1_ind, 'motion': motion1})])
                            time_b[bin1_ind][len(time_b[bin1_ind])] = {'key': key_2_search,
                                                                       'list': time_b[bin1_ind][specific_x]['list'] + [
                                                                           {'bin_diff': bin2_ind - bin1_ind,
                                                                            'motion': (bin2_ind, m2_i, 'all_fbp_a')}]}
                            specific_x = len(time_b[bin1_ind])-1

                        # all_fbp_a[bin2_ind][motion1] = None  # removing it for later iterates
                    # visited.append((bin2_ind, motion1))
    # choose which aggregations will be performed among the possible ones
    # to this end, shuffle the order in which the aggregations will be considered;
    # there must be at least 2 motioncodes to perform an aggregation over time.
    # There might be joint sets inside time aggregation which needs to be aggregated
    # inside the bins. otherwise just check the possible aggregation inside each bin as well.



    # Todo: We should also apply a time aggregation for those who are not aggregated after perform agg step
    # Todo: to connect alone motioncodes through time.

    # possible_aggregs = [('intptt', k) for k, v in intptt_a.items() if len(v) > 1] + \
    #                    [('fbp', k) for k, v in fbp_a.items() if len(v) > 1]

    # ------------------------------------------------------------
    # 3. Possible Aggregations
    possible_aggregs = []
    for bin_ind in range(len(motioncodes_1p)):
        # 1. over interpretations
        for k in time_a[bin_ind]:
            possible_flag = False
            v = time_a[bin_ind][k]
            # There are two types of Agg. over bins and inside each bin
            # Since we randomized the range, so we don't need to consider different combinations.
            # we combine them based on the elected length from the prev. step
            # If Agg. happens thrugh bins, we should include it in the possible Agg.
            # regardless of whether it needs insider Agg. or not. We later consider that Agg.
            # in the "perform Agg." step. Otherwise, we should also check if it needs an insider Agg.
            contributors = v['list']
            if len(contributors)>1: # Through time windows
                possible_flag = True

            if len(contributors)==1: # Inside the current window
                c_bin_ind = contributors[0]['bin_diff'] +  bin_ind# it only has one joint set we check inside time window
                c_intpt_a_key = contributors[0]['motion']
                m1_intrpts = all_intptt_a[c_intpt_a_key[0]][c_intpt_a_key[1]]['list']
                if len(m1_intrpts)>1: # if it's composed of more than one motion
                    possible_flag = True
            # Check joint set validity --> it is already considered in the prev.step
            # Seems it is not correct due to the second body part hashing??
            # Also, it doesn't cover all possiblities throgh windows due to %100 inside agg.

            if possible_flag:
                weight = len(contributors)*len(m1_intrpts) # We should also count plural joints twice
                possible_aggregs += [('intptt', {'bin': bin_ind, 'list': v['list']}, weight)]
                # --> all_intptt_a-->motioncodes_1p
                all_intptt_a
                motioncodes_1p, all_intptt_a

        # 2. Over joint-based
        for k in time_b[bin_ind]:
            possible_flag = False
            v = time_b[bin_ind][k]
            # There are two types of Agg. over bins and inside each bin
            # Since we randomized the range, so we don't need to consider different combinations.
            # we combine them based on the elected length from the prev. step
            # If Agg. happens thrugh bins, we should include it in the possible Agg.
            # regardless of whether it needs insider Agg. or not. We later consider that Agg.
            # in the "perform Agg." step. Otherwise, we should also check if it needs an insider Agg.
            contributors = v['list']
            if len(contributors) > 1: # Through time
                possible_flag = True

            if len(contributors) == 1: # Inside the current window
                c_bin_ind = contributors[0]['bin_diff'] + bin_ind  # it only has one joint set we check inside time window
                c_fbp_a_key = contributors[0]['motion']
                m1_intrpts = all_fbp_a[c_fbp_a_key[0]][c_fbp_a_key[1]]['list']
                if len(m1_intrpts) > 1:  # if it's composed of more than one motion
                    possible_flag = True

            if possible_flag:
                weight = len(contributors) * len(m1_intrpts) # Todo: plural cases not considered
                possible_aggregs += [('fbp', {'bin': bin_ind, 'list': contributors}, weight)]


    # We should double check the joints based on the focus body part to make sure we aggregate
    # motion codes with exactly the same joint set while some motion keys might be None in the
    # Possible_aggregs. **SOLVED**: in prev. step
    # We need to do it before appending to the possible_Aggregs since it might exclude some combinations
    # For example RElbow and RKnee will loose their chance to be appear in the eligible aggregs due to
    # presence of LElbow and LKnee with different motion

    random.shuffle(
        possible_aggregs)  # potential aggregations will be studied in random order, independently of their kind
    # Todo: it should be proportional to their weight
    aggregs_to_perform = []  # list of the aggregations to perform later (either intptt-based or fbp-based)
    unavailable_motion_inds = set()  # indices of the posecodes that will be aggregated
    for agg in possible_aggregs:
        # get the list of posecodes ids that would be involved in this aggregation

        # m_inds = all_intptt_a[agg[1]['bin']][agg[1]['key']] if agg[0] == "intptt" else all_fbp_a[agg[1]['bin']][agg[1]['key']]
        bin_ind = agg[1]['bin']
        m_time_a_inds = agg[1]['list']
        m_a_inds = []
        # for item in (m_time_a_inds):
        #     m_a_inds +=[all_intptt_a[item['motion'][0]][item['motion'][1]]['list']]
        try:
            if agg[0] == "intptt":
                m1_inds_required = [x  for item in m_time_a_inds for x in
                                    all_intptt_a[item['motion'][0]][item['motion'][1]]['list'][:]]
            if agg[0] == "fbp":
                m1_inds_required = [x for item in m_time_a_inds for x in
                                    all_fbp_a[item['motion'][0]][item['motion'][1]]['list'][:]]
        except:
            print()
        m1_inds_available = list(set(m1_inds_required).difference(unavailable_motion_inds))

        # m_inds = [ tuple([x[0], x[1]]) for x in m_inds] # we carried interpretations as the third element for debugging purpose.
        # check that all or a part of them are still available for aggregation
        # Note that, since we already consider all possible combinations, all joints should
        # be available, not partially.
        if len(m1_inds_required) == len(m1_inds_available):
            # m_inds = list(set(m_inds).difference(unavailable_motion_inds))
            if len(m1_inds_required) > 1:  # there must be at least 2 (unused, hence available) posecodes to perform an aggregation
                # update list of posecode indices to aggregate
                all_intptt_a
                # random.shuffle(m1_inds_required)  # shuffle to later aggregate these posecodes in random order inside bins
                if agg[0] == "intptt":
                    for agg_parts in agg[1]['list']:
                        m1_parts = all_intptt_a[agg_parts['motion'][0]][agg_parts['motion'][1]]['list']
                        random.shuffle(m1_parts)
                    # intptt_a[agg[1]]['list'] = 1 #m_inds
                    # all_intptt_a[agg[1]['list']['motion']] = m1_inds_required
                elif agg[0] == "fbp":
                    for agg_parts in agg[1]['list']:
                        m1_parts = all_fbp_a[agg_parts['motion'][0]][agg_parts['motion'][1]]['list']
                        random.shuffle(m1_parts)
                    # fbp_a[agg[1]] = 1 # m_inds
                # grant aggregation (to perform later)
                unavailable_motion_inds.update(m1_inds_required) #m_inds)
                aggregs_to_perform.append(agg)
                motioncodes_1p



    # perform the elected aggregations
    if extra_verbose: print("Aggregations to perform:", aggregs_to_perform)
    updated_motioncodes = [[] for _ in range(len(motioncodes_1p))]
    def extract_aggs(agg):
        subject_joints = []  # we should pick a set of uniques
        agg_interprets = []
        agg_object_joints = []
        for m_aggreg_inside in agg[1]['list']:
            try:
                # motion_inside = time_a[m_aggreg_inside['motion'][0]][m_aggreg_inside['motion'][1]]
                # motion_inside = all_intptt_a[m_aggreg_inside['motion'][0]][m_aggreg_inside['motion'][1]]
                # motion1_inside = all_intptt_a_ind['list']
                if agg[0] == 'fbp':
                    motion_inside = all_fbp_a[m_aggreg_inside['motion'][0]][m_aggreg_inside['motion'][1]]
                elif agg[0] == 'intptt':
                    motion_inside = all_intptt_a[m_aggreg_inside['motion'][0]][m_aggreg_inside['motion'][1]]
                else:
                    print()
            except:
                print()
            # m1_inds = [m1_[0:2] for m1_ in motion_inside['list']]
            m1_inds = [m1 for m1 in motion_inside['list']] # [m1_['motion'] for m1_ in motion_inside['list']]
            verb_adjective = []
            object_joints = []
            # for single_m_a_address in m1_inds:
            #     for single_m_address in all_intptt_a[single_m_a_address[0]][single_m_a_address[1]]['list']:
            if True:
                for single_m_address in m1_inds:
                    try:
                        single_m = motioncodes_1p[single_m_address[0]][single_m_address[1]]
                    except:
                        print('X#$%!')
                    subject_joints.append(tuple(single_m[:2]))
                    object_joints.append(tuple(single_m[3:]))
                    verb_adjective.append({'bin_diff': m_aggreg_inside['bin_diff'],
                                           'spatial': single_m[2]['spatial'],
                                           'temporal': single_m[2]['temporal']})
                if agg[0] == 'intptt': # TODO: how about if we have a super joint set with several interpretations Agg.?!?
                    interpret_candidate = random.choice(
                        verb_adjective)  # we know that spatials are equal but not sure about temporal
                    # so, we pick a random that might have temporal intpt as weel. TODO: I think this is incorrect
                    agg_interprets.append(interpret_candidate)

                    # at each step the objects should be the same, otherwise they don't make it to here'
                    object_candidate = [(side, body_part) for side, body_part in set(object_joints)]
                    # Ther result should be one joint ??
                    agg_object_joints.extend(object_candidate)
                elif agg[0] == 'fbp':
                    interpret_candidate = verb_adjective
                    agg_interprets.extend(verb_adjective)
                    # agg_interprets += (verb_adjective)
                    agg_object_joints.extend(object_joints)


                # This check is not appropriate, we shuld find a way to sanity check
                # maybe it is correct because all must have the same second body part
                # if len(object_candidate) != 1:
                #     print("len(object_candidate) != 1")


                # agg_object_joints += (object_joints)

        # This check is not appropriate, we shuld find a way to sanity check
        subject_candidate = set(subject_joints)
        # if len(subject_candidate) != len(subject_joints):
        #     raise NotImplementedError

        # perform the interpretation-based aggregation over time and window
        # agg[1]: (size 3) interpretation id, side2, body_part2
        subject_joints_unique = [[side, body_part] for side, body_part in set(subject_joints)]

        return {'subject_joints': subject_joints_unique, 'agg_interprets': agg_interprets, 'object_joints': agg_object_joints}

    covered_motioncodes = set()
    for agg in aggregs_to_perform:
        if random.random() < PROP_AGGREGATION_HAPPENS: # it should be adjust to be proportional to its weight
            ex_agg = extract_aggs(agg)
            if agg[0] == "intptt":
                # new_posecode = [MULTIPLE_SUBJECTS_KEY, [motioncodes_1p[p_ind][:2] for p_ind in intptt_a[agg[1]]]] + list(agg[1])
                # Sometimes we Agg. interpretations over time while we have one joint, we should adjust the
                # MULTIPLE_SUBJECTS_KEY to something like SINGLE_SUBJECTS_KEY
                # We might handle it at the conversion step.
                new_motioncode = [MULTIPLE_SUBJECTS_KEY,  ex_agg['subject_joints'] ] + [list(ex_agg['agg_interprets'])] + [ex_agg['object_joints']]
                try:
                    m1_inds_covered = [x for item in agg[1]['list'] for x in
                                        all_intptt_a[item['motion'][0]][item['motion'][1]]['list'][:]]
                except:
                    print()

            elif agg[0] == "fbp":
                # perform the focus-body-part-based aggregation
                # agg[1]: (size 2) side1, body_part1
                # new_posecode = [JOINT_BASED_AGGREG_KEY, list(agg['list']),
                #                 [motioncodes_1p[p_ind][2] for p_ind in fbp_a[agg[1]]], #
                #                 [motioncodes_1p[p_ind][3:] for p_ind in fbp_a[agg[1]]]]
                new_motioncode = [JOINT_BASED_AGGREG_KEY, ex_agg['subject_joints']] + [list(ex_agg['agg_interprets'])] + [ex_agg['object_joints']]
                try:
                    m1_inds_covered = [x for item in agg[1]['list'] for x in
                                       all_fbp_a[item['motion'][0]][item['motion'][1]]['list'][:]]
                except:
                    print()
                # if performing interpretation-fusion, it should happen here
                # ie. ['<joint_based_aggreg>', ['right', 'arm'], [16, 9, 15], [['left', 'arm'], ['left', 'arm'], ['left', 'arm']]],
                # which leads to "the right arm is behind the left arm, spread far apart from the left arm, above the left arm"
                # whould become something like "the right arm is spread far apart from the left arm, behind and above it"
                # CONDITION: the second body part is not None, and is the same for at least 2 interpretations
                # CAUTION: one should avoid mixing "it" words refering to BP2 with "it" words refering to BP1...

            related_posecodes = set()
            related_motioncodes = []
            for mcx in agg[1]['list']:
                if mcx['motion'][2] == 'all_fbp_a':
                    rl_mcs = all_fbp_a[mcx['motion'][0]][mcx['motion'][1]]['list']
                    related_motioncodes += [rl_mcs]
                if mcx['motion'][2] == 'all_intptt_a':
                    rl_mcs = all_intptt_a[mcx['motion'][0]][mcx['motion'][1]]['list']
                    related_motioncodes += [rl_mcs]
                rel_pc = [pcx for m1x in rl_mcs for pcx in motioncodes_1p[m1x[0]][m1x[1]][2]['posecode']]
                related_posecodes.update(rel_pc)
            new_motioncode.append(related_posecodes)

            updated_motioncodes[agg[1]['bin']].append(new_motioncode)
            covered_motioncodes.update(m1_inds_covered)

    if extra_verbose:
        print("Posecodes from interpretation/joint-based aggregations:")
        for m in updated_motioncodes:
            print(m)

    # don't forget to add all the posecodes that were not subject to these kinds
    # of aggregations
    # updated_posecodes.extend([p for p_ind, p in enumerate(motioncodes_1p) if p_ind not in unavailable_motion_inds])
    # We appliy this step using covered_motioncodes instead of unavailable_motion_inds since we might not aggregate
    # a specific one and still want to keep it in the final description. I also commented on the posescripot git
    # repo about it.
    for i in range(len(motioncodes_1p)):
        for j in range(len(motioncodes_1p[i])):
            if tuple([i, j, motioncodes_1p[i][j][2]['spatial'], 'point2M1']) not in covered_motioncodes:
            # if tuple([i, j, motioncodes_1p[i][j][2]['spatial'], 'point2M1']) not in unavailable_motion_inds:
                mc = motioncodes_1p[i][j]
                if not isinstance(mc[2], list):  # if not a result from the Agg. step
                    mc[2]['bin_diff'] = 0
                    mc = ['<SINGLE>', [[mc[0], mc[1]]], [mc[2]], [[mc[3], mc[4]]], set(mc[2]['posecode'])]
                updated_motioncodes[i].append(mc)

    if len(updated_motioncodes[0])==1:
        print()
    return updated_motioncodes

def infer_timecodeds(agg_motioncodes):


    # Here we immediately jump to skip and format since the
    # time queriies are already applied and provided by 'bin_diff'
    # key in each aggregated motioncodes
    # We should shuffle motioncodes inside each bin here rather than the next step

    very_first_motion_of_the_bin_skipped = False

    skipped = []
    t_operator = TIMECODE_OPERATORS["ChronologicalOrder"]
    last_timecourse = 0
    for win_ind in range(len(agg_motioncodes)):
        random.shuffle(agg_motioncodes[win_ind])

        for m_ind in range(len(agg_motioncodes[win_ind])):
            mc = agg_motioncodes[win_ind][m_ind]
            if not isinstance(mc[2], list):  # if not a result from the Agg. step
                mc[2]['bin_diff'] = 0
                mc = ['<SINGLE>', [[mc[0], mc[1]]], [mc[2]], [[mc[3], mc[4]] ]  ]


            # Calculating time states
            # Adjust the first bin w.r.t. the last motion
            current_timecourse = mc[2][0]['bin_diff'] + win_ind # The first one of the current window which is always 0
            mc[2][0]['bin_diff'] =  current_timecourse - last_timecourse
            # last_time_state = mc[2][-1]['bin_diff'] + win_ind   # The last one of the current window
            if len(mc[2])>1:
                # adjust the time timecourse w.r.t. the last motion of this chain
                last_timecourse = mc[2][-1]['bin_diff'] + win_ind
            else:
                # This is because the first bin is the only motion
                # happening in the current aggreagtion. So, when
                # len(mc[2])=1:
                #       We modified the bin_diff w.r.t the prev. motion and
                #       can't be used to determine the current timecourse
                #       at the end of the aggregation. Otherwise, the last
                #       bin is relative to the current window.
                last_timecourse = win_ind

            bin_diff_batch = np.array([x['bin_diff'] for x in mc[2]])
            # Here we should calssify each bin_diff

            t_values = t_operator.eval(bin_diff_batch)
            t_intptt = t_operator.interprete(t_values)

            # Eligibility
            general_time_query = GENERAL_TIMECODES[0]
            general_t_intptt_a = general_time_query[2] if general_time_query[2] != [] \
                else [INTPTT_NAME2ID_TIME[x] for x in TIMECODE_OPERTATOR_VALUES['ChronologicalOrder']['category_names']]
            general_t_intptt_r = [INTPTT_NAME2ID_TIME[x] for x in general_time_query[3]]

            t_elig = np.zeros_like(t_intptt)
            for t in range(len(t_intptt)):
                t_elig[t] =(1 if t_intptt[t] in general_t_intptt_a else 0) + (
                       1 if t_intptt[t] in general_t_intptt_r else 0)


                # Skip and format





                random_skip = True
                if (t_elig[t] == 2) or \
                        (t_elig[t] and (not random_skip or random.random() >= PROP_SKIP_TIMECODE)):
                    mc[2][t]['chronological_order'] = t_intptt[t] # time interpretation index


                elif random_skip and t_elig[t]:
                    skipped.append((win_ind, m_ind, t, t_intptt[t]))
                    mc[2][t]['chronological_order'] = None
                else:
                    mc[2][t]['chronological_order'] = None

                # EXCEPTION: for the first sentence we only let rare time
                # be eligible and make it to the description, i.e. the
                # begining of the clip is stationary
                if not very_first_motion_of_the_bin_skipped and t_elig[t] < 2:
                    mc[2][t]['chronological_order'] = None
                very_first_motion_of_the_bin_skipped = True

            #     No Agg. Step.
            # We are done here and decided about time relations
    return agg_motioncodes




            # Put it for the eligble ones considering the start of the sentence.
    raise NotImplementedError

################################################################################
## CONVERT POSECODES, POLISHING STEP
################################################################################

def side_and_plural(side, determiner="the"):
    if side is None:
        return f'{determiner}', 'is'
    if side == PLURAL_KEY:
        return random.choice(["both", determiner]), "are"
    else:
        return f'{determiner} {side}', 'is'


def side_body_part_to_text(side_body_part, determiner="the", new_sentence=False):
    """Convert side & body part info to text, and give verb info
    (singular/plural)."""
    # don't mind additional spaces, they will be trimmed at the very end
    side, body_part = side_body_part
    if side == JOINT_BASED_AGGREG_KEY:
        # `body_part` is in fact a list [side_1, true_body_part_1]
        side, body_part = body_part
    if side is None and body_part is None:
        return None, None
    if side == MULTIPLE_SUBJECTS_KEY:
        # `body_part` is in fact a list of sublists [side, true_body_part]
        sbp = [f"{side_and_plural(s, determiner)[0]} {b if b else ''}" for s,b in body_part]
        return f"{', '.join(sbp[:-1])} and {sbp[-1]}", "are"
    if body_part == "body":
        # choose how to refer to the body (general stance)
        if new_sentence:
            bp  = random.choice(SENTENCE_START).lower()
        else:
            bp = random.choice(BODY_REFERENCE_MID_SENTENCE).lower()
        # correction in particular cases
        if bp == "they":
            if determiner == "his": return "he", "is"
            elif determiner == "her": return "she", "is"
            return "they", "are"
        elif bp == "the human":
            if determiner == "his": return "the man", "is"
            elif determiner == "her": return "the woman", "is"
            return "the human", "is"
        elif bp == "the body":
            return f"{determiner} body", "is"
        return bp, "is"
    else:
        s, v = side_and_plural(side, determiner)
        return f"{s} {body_part if body_part else ''}", v


def omit_for_flow(bp1, verb, intptt_name, bp2, bp1_initial):
    """Apply some simple corrections to the constituing elements of the
    description piece to be produced for the sake of flow."""
    # remove the second body part in description when it is not necessary and it
    # simply makes the description more cumbersome
    if bp2 is None: bp2 = '' # temporary, to ease code reading (reset to None at the end)
    # hands/feet are compared to the torso to know whether they are in the back
    if 'torso' in bp2: bp2 = ''
    # hands are compared to their respective shoulder to know whether they are
    # out of line
    if 'hand' in bp1_initial and 'shoulder' in bp2 and intptt_name in ['at_right', 'at_left']: bp2 = ''
    # feet are compared to their respective hip to know whether they are out of line
    if 'foot' in bp1_initial and 'hip' in bp2 and intptt_name in ['at_right', 'at_left']: bp2 = ''
    # hands/wrists are compared with the neck to know whether they are raised high
    if ('hand' in bp1_initial or 'wrist' in bp1_initial) and 'neck' in bp2 and intptt_name == 'above': bp2 = ''
    return None if bp2=='' else bp2


def insert_verb(d, v):
    if v == NO_VERB_KEY:
        # consider extra-spaces around words to be sure to target them exactly
        # as words and not n-grams
        d = d.replace(" are ", " ").replace(" is ", " ")
        v = "" # to further fill the '%s' in the template sentences 
    # if applicable, try to insert verb v in description template d
    try :
        return d % v
    except TypeError: # no verb placeholder
        return d


def posecode_to_text(bp1, verb, intptt_id, bp2, bp1_initial, simplified_captions=False):
    """ Stitch the involved body parts and the interpretation into a sentence.
    Args:
        bp1 (string): text for the 1st body part & side.
        verb (string): verb info (singular/plural) to adapt description.
        inptt_id (integer): interpretation id
        bp2 (string): same as bp1 for the second body part & side. Can be None.
        bp1_initial (string): text for the initial 1st body part & side
                (useful if the provided bp1 is actually a transition text),
                useful to apply accurate text patches).
    Returns:
        string
    """
    intptt_name = INTERPRETATION_SET[intptt_id]
    # First, some patches
    if not simplified_captions:
        bp2 = omit_for_flow(bp1, verb, intptt_name, bp2, bp1_initial)
    # if the NO_VERB_KEY is found in bp1, remove the verb from the template
    # sentence (ie. replace it with "")
    if NO_VERB_KEY in bp1:
        bp1, verb = bp1[:-len(NO_VERB_KEY)], NO_VERB_KEY
    # Eventually fill in the blanks of the template sentence for the posecode
    if bp2 is None:
        # there is not a second body part involved
        try:
            d = random.choice(ENHANCE_TEXT_1CMPNT[intptt_name])
            d = d.format(bp1)
        except:
            d=''
    else:
        d = random.choice(ENHANCE_TEXT_2CMPNTS[intptt_name])
        d = d.format(bp1, bp2)
    d = insert_verb(d, verb)
    return d


def convert_posecodes(posecodes, simplified_captions=False, verbose=True):
    
    nb_poses = len(posecodes)
    nb_actual_empty_description = 0

    # 1) Produce pieces of text from posecodes
    descriptions = ["" for p in range(nb_poses)]
    determiners = ["" for p in range(nb_poses)]
    for p in range(nb_poses):

        # find empty descriptions
        if len(posecodes[p]) == 0:
            nb_actual_empty_description += 1
            # print(f"Nothing to describe for pose {p}.")
            continue # process the next pose

        # Preliminary decisions (order, determiner, transitions)
        # shuffle posecodes to provide pose information in no particular order
        random.shuffle(posecodes[p])
        # randomly pick a determiner for the description
        determiner = random.choices(DETERMINERS, weights=DETERMINERS_PROP)[0]
        determiner = DETERMINER_KEY
        determiners[p] = determiner
        # select random transitions (no transition at the beginning)
        # transitions = [""] + random.choices(TEXT_TRANSITIONS, TEXT_TRANSITIONS_PROP, k = len(posecodes[p]) - 1)
        transitions = [""] + random.choices(TEXT_TRANSITIONS, TEXT_TRANSITIONS_PROP,  k=len(posecodes[p]) - 1)
        with_in_same_sentence = False # when "with" is used as transition in a previous part of the sentence, all following parts linked by some particular transitions must respect a no-verb ('is'/'are') grammar

        # Convert each posecode into a piece of description
        # and iteratively concatenate them to the description 
        for i_pc, pc in enumerate(posecodes[p]):

            # Infer text for the first body part & verb
            bp1_initial, verb = side_body_part_to_text(pc[:2], determiner, new_sentence=(transitions[i_pc] in ["", ". "]))
            # Grammar modifications are to be expected if "with" was used as transition
            if transitions[i_pc] == ' with ' or \
                (with_in_same_sentence and transitions[i_pc] == ' and '):
                bp1_initial += NO_VERB_KEY
                with_in_same_sentence = True
            elif with_in_same_sentence and transitions[i_pc] != ' and ':
                with_in_same_sentence = False

            # Infer text for the secondy body part (no use to catch the verb as
            # this body part is not the subject of the sentence, hence the [0]
            # after calling side_body_part_to_text, this time)
            if pc[0] == JOINT_BASED_AGGREG_KEY:
                # special case for posecodes modified by the joint-based
                # aggregation rule
                # gather the names for all the second body parts involved
                bp2s = [side_body_part_to_text(bp2, determiner)[0] for bp2 in pc[3]]
                # create a piece of description fore each aggregated posecode
                # and link them together
                d = ""
                bp1 = bp1_initial
                special_trans = ". They " if verb=="are" else ". It " # account for a first body part that is plural (eg. the hands)
                for intptt_id, bp2 in zip(pc[2], bp2s):
                    d += posecode_to_text(bp1, verb, intptt_id, bp2, bp1_initial,
                                        simplified_captions=simplified_captions)
                    # choose the next value for bp1 (transition text)
                    if bp1 != " and ":
                        choices = [" and "+NO_VERB_KEY, special_trans, ", "+NO_VERB_KEY]
                        if NO_VERB_KEY not in bp1: choices += [" and "] 
                        bp1 = random.choice(choices)
                    else:
                        bp1 = special_trans

            else:
                bp2 = side_body_part_to_text(pc[3:5], determiner)[0] # default/initialization
                
                # If the two joints are the same, but for their side, choose at
                # random between:
                # - keeping the mention of the whole second body part
                #   (side+joint name),
                # - using "the other" to refer to the second joint as the
                #   first's joint counterpart,
                # - or simply using the side of the joint only (implicit
                #   repetition of the joint name)
                if not simplified_captions and pc[1] == pc[4] and pc[0] != pc[3]:
                    # pc[3] cannot be None since pc[1] and pc[4] must be equal (ie.
                    # these are necessarily sided body parts)
                    bp2 = random.choice([bp2, "the other", f"the {pc[3]}"])

                # Create the piece of description corresponding to the posecode
                d = posecode_to_text(bp1_initial, verb, pc[2], bp2, bp1_initial,
                                        simplified_captions=simplified_captions)
            
            # Concatenation to the current description
            descriptions[p] += transitions[i_pc] + d

        descriptions[p] += "." # end of the description
        
        # Correct syntax (post-processing)
        # - removing wide spaces,
        # - replacing "upperarm" by "upper arm"
        # - randomly replacing all "their"/"them" by "his/him" or "her/her" depending on the chosen determiner
        # - capitalizing when beginning a sentence
        descriptions[p] = re.sub("\s\s+", " ", descriptions[p])
        descriptions[p] = descriptions[p].replace("upperarm", "upper arm")
        descriptions[p] = '. '.join(x.capitalize() for x in descriptions[p].split('. '))
        if determiner in ["his", "her"]:
            # NOTE: do not replace "they" by "he/she" as "they" can sometimes
            # refer to eg. "the hands", "the feet" etc.
            # Extra-spaces allow to be sure to treat whole words only
            descriptions[p] = descriptions[p].replace(" their ", f" {determiner} ")
            descriptions[p] = descriptions[p].replace("Their ", f"{determiner}".capitalize()) # with the capital letter
            descriptions[p] = descriptions[p].replace(" them ", " him " if determiner=="his" else f" {determiner} ")

    if verbose: 
        print(f"Actual number of empty descriptions: {nb_actual_empty_description}.")

    return descriptions, determiners


def convert_posecodes4motioncode(posecodes, bp1_initial_input, simplified_captions=False, verbose=True):
    nb_poses = len(posecodes)
    nb_actual_empty_description = 0

    # 1) Produce pieces of text from posecodes
    descriptions = ["" for p in range(nb_poses)]
    determiners = ["" for p in range(nb_poses)]
    for p in range(nb_poses):

        # find empty descriptions
        if len(posecodes[p]) == 0:
            nb_actual_empty_description += 1
            # print(f"Nothing to describe for pose {p}.")
            continue  # process the next pose

        # Preliminary decisions (order, determiner, transitions)
        # shuffle posecodes to provide pose information in no particular order
        random.shuffle(posecodes[p])
        # randomly pick a determiner for the description
        determiner = random.choices(DETERMINERS, weights=DETERMINERS_PROP)[0]
        determiner = DETERMINER_KEY
        determiners[p] = determiner
        # select random transitions (no transition at the beginning)
        # transitions = [""] + random.choices(TEXT_TRANSITIONS, TEXT_TRANSITIONS_PROP, k = len(posecodes[p]) - 1)
        transitions = [""] + random.choices(TEXT_TRANSITIONS, TEXT_TRANSITIONS_PROP, k=len(posecodes[p]) - 1)
        with_in_same_sentence = False  # when "with" is used as transition in a previous part of the sentence, all following parts linked by some particular transitions must respect a no-verb ('is'/'are') grammar

        # Convert each posecode into a piece of description
        # and iteratively concatenate them to the description
        for i_pc, pc in enumerate(posecodes[p]):

            # Infer text for the first body part & verb
            bp1_initial, verb = side_body_part_to_text(pc[:2], determiner,
                                                       new_sentence=(transitions[i_pc] in ["", ". "]))

            if bp1_initial_input:
                bp1_initial=bp1_initial_input # it happens when only one posecode is involved

            # Grammar modifications are to be expected if "with" was used as transition
            if transitions[i_pc] == ' with ' or \
                    (with_in_same_sentence and transitions[i_pc] == ' and '):
                bp1_initial += NO_VERB_KEY
                with_in_same_sentence = True
            elif with_in_same_sentence and transitions[i_pc] != ' and ':
                with_in_same_sentence = False

            # Infer text for the secondy body part (no use to catch the verb as
            # this body part is not the subject of the sentence, hence the [0]
            # after calling side_body_part_to_text, this time)
            if pc[0] == JOINT_BASED_AGGREG_KEY:
                # special case for posecodes modified by the joint-based
                # aggregation rule
                # gather the names for all the second body parts involved
                bp2s = [side_body_part_to_text(bp2, determiner)[0] for bp2 in pc[3]]
                # create a piece of description fore each aggregated posecode
                # and link them together
                d = ""
                bp1 = bp1_initial
                special_trans = ". They " if verb == "are" else ". It "  # account for a first body part that is plural (eg. the hands)
                for intptt_id, bp2 in zip(pc[2], bp2s):
                    d += posecode_to_text(bp1, verb, intptt_id, bp2, bp1_initial,
                                          simplified_captions=simplified_captions)
                    # choose the next value for bp1 (transition text)
                    if bp1 != " and ":
                        choices = [" and " + NO_VERB_KEY, special_trans, ", " + NO_VERB_KEY]
                        if NO_VERB_KEY not in bp1: choices += [" and "]
                        bp1 = random.choice(choices)
                    else:
                        bp1 = special_trans

            else:
                bp2 = side_body_part_to_text(pc[3:5], determiner)[0]  # default/initialization

                # If the two joints are the same, but for their side, choose at
                # random between:
                # - keeping the mention of the whole second body part
                #   (side+joint name),
                # - using "the other" to refer to the second joint as the
                #   first's joint counterpart,
                # - or simply using the side of the joint only (implicit
                #   repetition of the joint name)
                if not simplified_captions and pc[1] == pc[4] and pc[0] != pc[3]:
                    # pc[3] cannot be None since pc[1] and pc[4] must be equal (ie.
                    # these are necessarily sided body parts)
                    bp2 = random.choice([bp2, "the other", f"the {pc[3]}"])

                # Create the piece of description corresponding to the posecode
                d = posecode_to_text(bp1_initial, verb, pc[2], bp2, bp1_initial,
                                     simplified_captions=simplified_captions)

            # Concatenation to the current description
            descriptions[p] += transitions[i_pc] + d

        # descriptions[p] += "."  # end of the description --> no need
    '''
        # Correct syntax (post-processing)
        # - removing wide spaces,
        # - replacing "upperarm" by "upper arm"
        # - randomly replacing all "their"/"them" by "his/him" or "her/her" depending on the chosen determiner
        # - capitalizing when beginning a sentence
        descriptions[p] = re.sub("\s\s+", " ", descriptions[p])
        descriptions[p] = descriptions[p].replace("upperarm", "upper arm")
        descriptions[p] = '. '.join(x.capitalize() for x in descriptions[p].split('. '))
        if determiner in ["his", "her"]:
            # NOTE: do not replace "they" by "he/she" as "they" can sometimes
            # refer to eg. "the hands", "the feet" etc.
            # Extra-spaces allow to be sure to treat whole words only
            descriptions[p] = descriptions[p].replace(" their ", f" {determiner} ")
            descriptions[p] = descriptions[p].replace("Their ", f"{determiner}".capitalize())  # with the capital letter
            descriptions[p] = descriptions[p].replace(" them ", " him " if determiner == "his" else f" {determiner} ")
    '''
    if verbose:
        print(f"Actual number of empty descriptions: {nb_actual_empty_description}.")

    return descriptions, determiners


################################################################################
## CONVERT MOTIONCODEs, POLISHING STEP
################################################################################
def insert_verb_motion(d, v, prev_tense, velocity, timerelation, pose_info):
    # if v == NO_VERB_KEY:
    #     # consider extra-spaces around words to be sure to target them exactly
    #     # as words and not n-grams
    #     d = d.replace(" are ", " ").replace(" is ", " ")
    #     v = "" # to further fill the '%s' in the template sentences
    # if applicable, try to insert verb v in description template d

    if prev_tense == 'None':
        if v == NO_VERB_SINGULAR_KEY or v == NO_VERB_PLURAL_KEY:
            d = d.replace(" are ", " ").replace(" is ", " ")
            tense = 's' if v == NO_VERB_SINGULAR_KEY else ''
            v = ''
        # if v == "":
        #     tense = 's'
        elif random.random() > 0.5:
            if v == "is":
                v = ""
                tense = "s"
            elif v == "are":
                v = ""
                tense = ""
        else:
            tense = "ing"
    else:
        if v == NO_VERB_SINGULAR_KEY or v == NO_VERB_PLURAL_KEY:
            d = d.replace(" are ", " ").replace(" is ", " ")
            v = ''
            tense = 's' if v == NO_VERB_SINGULAR_KEY else ''
            tense = prev_tense
        elif (prev_tense=='' or prev_tense=='s'):
            if v == "is":
                v = ""
                tense = "s"
            elif v == "are":
                v = ""
                tense = ""
        elif prev_tense=='ing':
            tense = "ing"



    # Before this step we should check if the last letter is "e" or "t" or "s" and so forth: Fixed in post-processing
    d = d.replace(VERB_TENSE, tense)

    if timerelation != '':
        timerelation = timerelation +', '
    d = d.replace(TIME_RELATION_TERM, timerelation)

    # ablation:
    # velocity = ''
    if velocity == '':
        d = d.replace(VELOCITY_TERM, '')
        d = d.replace(AND_VELOCITY_TERM, '')
    else:
        # Usually for the moderate velocity
        velocity_adjective = random.choice(VELOCITY_ADJECTIVES[velocity])
        if velocity_adjective != '':
            d = d.replace(VELOCITY_TERM, ' '+ velocity_adjective)
            d = d.replace(AND_VELOCITY_TERM, ' and ' + velocity_adjective)
        else:
            d = d.replace(VELOCITY_TERM, '')
            d = d.replace(AND_VELOCITY_TERM, '')

    # Pose state:
    if pose_info[0] == 'INITIAL_STATE':
        pose_info[1] = pose_info[1] + random.choice(pose2action_transitions)
        d = d.replace(INITIAL_POSE_TERM, pose_info[1])
    if pose_info[0] == 'FINAL_STATE':
        pose_info[1] = random.choice([', ', '. ', ', and ']) + random.choice(action2pose_transitions) + ' ' + pose_info[1]
        d = d.replace(FINAL_POSE_TERM, pose_info[1])
    d = d.replace(INITIAL_POSE_TERM, '')
    d = d.replace(FINAL_POSE_TERM, '')

    try :
        return d % v, tense
    except TypeError: # no verb placeholder
        return d, tense
def motioncode_to_text(bp1, verb, prev_tense, intptt_info, bp2, bp1_initial, timerelation_intptt, pose_info, simplified_captions=False):
    """ Stitch the involved body parts and the interpretation into a sentence.
    Args:
        bp1 (string): text for the 1st body part & side.
        verb (string): verb info (singular/plural) to adapt description.
        inptt_id (integer): interpretation id
        bp2 (string): same as bp1 for the second body part & side. Can be None.
        bp1_initial (string): text for the initial 1st body part & side
                (useful if the provided bp1 is actually a transition text),
                useful to apply accurate text patches).
    Returns:
        string
    """
    intptt_name_spatial = INTERPRETATION_SET_MOTION[intptt_info['spatial']]
    intptt_name_temporal = INTERPRETATION_SET_MOTION[intptt_info['temporal']] if intptt_info['temporal'] is not None else ''

    time_relation_key = INTERPRETATION_SET_TIME[timerelation_intptt] \
        if timerelation_intptt is not None else ''
    time_relation_value = random.choice(
        CHRONOLOGICAL_ORDER_ADJECTIVE[time_relation_key]) if time_relation_key is not '' else ''
    # First, some patches
    if not simplified_captions:
        bp2 = omit_for_flow(bp1, verb, intptt_name_spatial, bp2, bp1_initial)
    # if the NO_VERB_KEY is found in bp1, remove the verb from the template
    # sentence (ie. replace it with "")
    if NO_VERB_KEY in bp1:
        # bp1, verb = bp1[:-len(NO_VERB_KEY)], NO_VERB_KEY
        bp1  = bp1[:-len(NO_VERB_KEY)]
        verb = NO_VERB_SINGULAR_KEY if verb == 'is' else NO_VERB_PLURAL_KEY

    # Angelica suggested to remove present continuous
    verb = NO_VERB_SINGULAR_KEY if verb == 'is' else NO_VERB_PLURAL_KEY

    # Here we decide whether we want to also include corresponding
    # posecodes or not. We should od it here since it might affect the
    # template selection process
    # PROBABLITY_OF_ADDING_POSECODE = 95
    # if random.random() < PROBABLITY_OF_ADDING_POSECODE:
    #     pose_code_description = ""
    #     # 1. Find candidate aggregated posecodes that covers our related posecodes list
    #
    # Eventually fill in the blanks of the template sentence for the posecode
    try:
        if bp2 is None:
            # there is not a second body part involved

                d = random.choice(ENHANCE_TEXT_1CMPNT_Motion[intptt_name_spatial])
                d = d.format(bp1)

        else:
            d = random.choice(ENHANCE_TEXT_2CMPNT_Motion[intptt_name_spatial])
            # d = d.format(bp1, bp2) # due to some rules with object joint twice
            d = d.format(bp1)
            d = d.replace(sj_obj, bp2)
    except:
        d=''
    # Some step:


    d, used_tense = insert_verb_motion(d, verb, prev_tense, intptt_name_temporal, time_relation_value, pose_info)
    return d, used_tense

def convert_motioncodes(posecodes, motioncodes, time_bin_info, simplified_captions=False, verbose=True):

    # nb_poses = len(posecodes)
    nb_windows = len(motioncodes)
    nb_actual_empty_description = 0


    # 1) Produce pieces of text from posecodes

    descriptions = ["" for win_ind in range(nb_windows)]
    determiners = ["" for win_ind in range(nb_windows)]
    global_determiner = random.choices(DETERMINERS, weights=DETERMINERS_PROP)[0]
    global_determiner = DETERMINER_KEY
    for win_ind in range(nb_windows):
        nb_motions = len(motioncodes[win_ind])
        # find empty descriptions
        if len(motioncodes[win_ind]) == 0:
            nb_actual_empty_description += 1
            # print(f"Nothing to describe for pose {p}.")
            continue  # process the next pose

        # Preliminary decisions (order, determiner, transitions)
        # shuffle motioncodes to provide motion information in no particular order
        # random.shuffle(motioncodes[win_ind]) commented because we did it in the infer_time_step
        # randomly pick a determiner for the description
        determiner = random.choices(DETERMINERS, weights=DETERMINERS_PROP)[0]
        determiner = DETERMINER_KEY
        determiners[win_ind] = determiner

        # select random transitions (no transition at the beginning)
        transitions = [""] + random.choices(TEXT_TRANSITIONS_With_TIME, k=len(motioncodes[win_ind]) - 1)

        # Since in each set of sentences we would like to describe one idea and avoid run-on sentences we set all transition to '. '
        transitions = [""] + [". "] *  (len(motioncodes[win_ind])-1)
        word_count_since_period = 0
        with_in_same_sentence = False  # when "with" is used as transition in a previous part of the sentence,
        # all following parts linked by some particular transitions must respect a no-verb ('is'/'are') grammar

        # Convert each motioncode into a piece of description
        # and iteratively concatenate them to the description inside windows
        for i_mc, mc in enumerate(motioncodes[win_ind]):
            if not isinstance(mc[2], list): # if not a result from the Agg. step
                mc[2]['bin_diff'] = 0
                mc = ['<SINGLE>', [[mc[0], mc[1]]], [mc[2]], [[mc[3], mc[4]]] ]
            if mc[4] == {('orientation_roll', 0)}:
                print()
            # Infer text for the first body part & verb
            if len(mc[1])>1:
                bp1_initial, verb = side_body_part_to_text([MULTIPLE_SUBJECTS_KEY, mc[1]], determiner,
                                       new_sentence=False) #new_sentence=(transitions[i_mc] in ["", ". "]))
            else: # when we have only one subject, to prevent wrong VERB
                bp1_initial, verb = side_body_part_to_text(mc[1][0], determiner,
                                                   new_sentence=False) # new_sentence=(transitions[i_mc] in ["", ". "]))

            # Grammar modifications are to be expected if "with" was used as transition
            if transitions[i_mc] == ' with ' or \
                    (with_in_same_sentence and transitions[i_mc] == ' and '):
                bp1_initial += NO_VERB_KEY
                with_in_same_sentence = True
            elif with_in_same_sentence and transitions[i_mc] != ' and ':
                with_in_same_sentence = False

            # Infer text for the second body part (no use to catch the verb as
            # this body part is not the subject of the sentence, hence the [0]
            # after calling side_body_part_to_text, this time)

            # infer posecodes:
            required_pose_set = mc[4]

            # This could be only 0, -1 since otherwise, it makes the transition complicated
            # Also, thanks to our randomization at the aggregation step, we do not need to consider
            # middle pose states.
            # Revise: We only consider the initial position of the joints
            # and we do not consider references due to subject/object

            # posecode_where_ind =  random.choice(mc[2])['bin_diff']
            # posecode_where_ind = random.choice([0, -1])

            # posecode_start_end_pos = random.choice(['INITIAL_STATE', 'FINAL_STATE'])
            posecode_start_end_pos = 'INITIAL_STATE'

            # For the Ablation study on pose injection
            # posecode_start_end_pos = 'FINAL_STATE'
            posecode_start_end_pos = 'NO_POSE_INJECTION'
            posecode_start_end_pos = 'INITIAL_STATE'

            # Check if spatial relation posecode
            # We remove them since it is already included in the spatial
            # motioncodes implicitly.
            for x in copy.deepcopy(required_pose_set):
                if x[0] in ['relativePosX', 'relativePosY', 'relativePosZ']:
                    required_pose_set.remove(x)



            # if posecode_where_ind == 0: # its bin_diff might be affected w.r.t. the prev. motioncode
            # #while it is actuallly belongs to the current window
            #     target_frame = (win_ind + 0
            #                     if posecode_start_end_pos == 'INITIAL_STATE' else 1) * time_bin_info['bin_size']
            # else:
            #     target_frame = (win_ind + mc[2][-1]['bin_diff']
            #                     if posecode_start_end_pos == 'INITIAL_STATE' else 1) * time_bin_info['bin_size']
            #

            # Since we decided to put only the initial state, and it always would
            # be the state of the joints in the current window regardless of its
            # time-relationship to the previous motioncode.
            target_frame = win_ind * time_bin_info['bin_size']

            applicable_list = []
            pc_covered = []
            for pc_ind in range(len(posecodes[target_frame])):
                pc = posecodes[target_frame][pc_ind]
                if pc[0]=='<joint_based_aggreg>':
                    pc1cover = set(pc[4])
                else:
                    pc1cover = set(pc[5])
                pc_covered.append(pc1cover)


            selected_samples, selected_indices = min_samples_to_cover(required_pose_set, pc_covered)

            applicable_list = [posecodes[target_frame][index] for index in selected_indices]

            # convert applicable to sentence
            # posecode_descriptions, posecode_determiners = convert_posecodes([applicable_list], simplified_captions)


            # for bp1 = bp1_initial or not
            exact_coverage = True if len(set().union(*selected_samples)) == len(required_pose_set) else False

            if len(applicable_list)>0 and posecode_start_end_pos!='NO_POSE_INJECTION':
                posecode_flag = random.choice(INITIAL_STATE_TRANSITIONS) if posecode_start_end_pos == 'INITIAL_STATE'\
                    else random.choice(FINAL_STATE_TRANSITIONS)
                and_required = random.choice([True, False])
                special_trans = " they " if verb == "are" else " it "
                bp1_initial_to_poscecodes = None
                if posecode_start_end_pos == 'INITIAL_STATE':
                    # if exact_coverage:
                    #     bp1_initial = random.choice(['' + NO_VERB_KEY, special_trans, ', ' + NO_VERB_KEY, ])
                    # #     doesnt work for plural and probably entity based ones when two joints boils down to one

                    posecode_descriptions, posecode_determiners = convert_posecodes4motioncode([applicable_list],
                                                                                               bp1_initial_to_poscecodes,
                                                                                               simplified_captions)
                    d_pose = '' + posecode_flag + "".join(posecode_descriptions) + random.choice([', and ', '. ', ' and '])

                    if len("".join(posecode_descriptions))< 5:
                        print("What?")
                else:
                    # if exact_coverage and len(applicable_list)==1:
                    #     # here we should somehow determine if we want to use special term for posecode description
                    #     bp1_initial_to_poscecodes = special_trans

                    posecode_descriptions, posecode_determiners = convert_posecodes4motioncode([applicable_list],
                                                                                               bp1_initial_to_poscecodes,
                                                                                               simplified_captions)
                    # d_pose = random.choice([', ', '. ', ', and ']) + posecode_flag + "".join(posecode_descriptions)
                    d_pose = '' + posecode_flag + "".join(posecode_descriptions)
            else:
                d_pose = ''
            # d_pose = d_pose.strip()


            if mc[0] == JOINT_BASED_AGGREG_KEY:
                if False:

                    # special case for posecodes modified by the joint-based
                    # aggregation rule
                    # gather the names for all the second body parts involved
                    bp2s = [side_body_part_to_text(bp2[0], determiner)[0] for bp2 in mc[3]]
                    # create a piece of description fore each aggregated posecode
                    # and link them together
                    d = ""
                    bp1 = bp1_initial
                    special_trans = ". They " if verb == "are" else ". It "  # account for a first body part that is plural (eg. the hands)
                    for intptt_id, bp2 in zip(mc[2], bp2s):
                        d += posecode_to_text(bp1, verb, intptt_id, bp2, bp1_initial,
                                              simplified_captions=simplified_captions)
                        # choose the next value for bp1 (transition text)
                        if bp1 != " and ":
                            choices = [" and " + NO_VERB_KEY, special_trans, ", " + NO_VERB_KEY]
                            if NO_VERB_KEY not in bp1: choices += [" and "]
                            bp1 = random.choice(choices)
                        else:
                            bp1 = special_trans

            # else:
            if True:
                # bp2 = side_body_part_to_text(mc[3:5], determiner)[0]  # default/initialization
                # bp2
                if not isinstance(mc[2], list):
                    mc = ['<SINGLE>', [mc[0], mc[1]], [mc[2]], [mc[3], mc[4]]]

                bp1 = bp1_initial
                # special_trans = ". They " if verb == "are" else ". It "  # account for a first body part that is plural (eg. the hands)
                special_trans = " they " if verb == "are" else " it "  # account for a first body part that is plural (eg. the hands)
                if 'body' in bp1_initial:
                    special_trans = "<s/he>"
                    bp1_initial = "<s/he>"
                    bp1 = bp1_initial


                inner_transition = transitions[i_mc] # ???
                inner_transition = [""] + random.choices(TEXT_TRANSITIONS_With_TIME, k=len(mc[2]) - 1)
                inner_transition = '' # the first one will be filled with the outer transition
                d_aggregated = ''
                prev_tense = 'None'

                #TODO Based on my discussion:
                # 1. No sentences longer than a specific length.
                # 2. No pronoun repetition inside an aggregated sentences.
                # 3. Repeat the pronoun every few sentences.
                # We consider a word counter to put period at the end
                # word_count_since_period = 0 Moved to the upper loop because of outer transitions (transitions)
                word_count_since_bp1initial = 0

                for i_magg in range(len(mc[2])):
                    # bp2 = side_body_part_to_text(mc[3][i_magg], determiner)[0]  # default/initialization
                    try:
                        # bp2 = side_body_part_to_text(mc[3][i_magg][0], determiner)[0] # zero because it would be always one joint for intrpt Agg.s
                        bp2 = side_body_part_to_text(mc[3][i_magg]   , determiner)[0]  # zero because it would be always one joint for intrpt Agg.s
                    except:
                        print("ZX! Fix this by making a list of intptr and joints in the agg or add if here")
                        cond = mc[0] == JOINT_BASED_AGGREG_KEY
                        mc = ['<SINGLE>', [mc[0], mc[1]], [mc[2]], [mc[3], mc[4]]]

                        bp2 = side_body_part_to_text(mc[3][i_magg][0], determiner)[0]
                    # If the two joints are the same, but for their side, choose at
                    # random between:
                    # - keeping the mention of the whole second body part
                    #   (side+joint name),
                    # - using "the other" to refer to the second joint as the
                    #   first's joint counterpart,
                    # - or simply using the side of the joint only (implicit
                    #   repetition of the joint name)

                    # if not simplified_captions and mc[1][0][1] == mc[3][i_magg][0][1] and mc[1][0][0] != mc[3][i_magg][0][0]: # due to equality of second part in Agg based
                    if not simplified_captions and mc[1][0][1] == mc[3][i_magg][1] and mc[1][0][0] != mc[3][i_magg][0]: # probably we also need some len checking due to several joints.
                    # if not simplified_captions and pc[1] == pc[4] and pc[0] != pc[3]:
                        # pc[3] cannot be None since pc[1] and pc[4] must be equal (ie.
                        # these are necessarily sided body parts)
                        bp2 = random.choice([bp2, "the other one", f"the {mc[3][i_magg][0]} one"])
                        bp2 = f"the {mc[3][i_magg][0]} one"

                    # Create the piece of description corresponding to the posecode
                    # TODO: We may add starting and ending pose states that are involved in the motion from the posescript
                    # It should be here since we want to only include posecodes once maybe at the begining or end

                    if  ((posecode_start_end_pos == 'INITIAL_STATE' and i_magg==0) or \
                        (posecode_start_end_pos == 'FINAL_STATE' and i_magg==len(mc[2])-1))\
                            and d_pose!='':
                        pose_state_info = [posecode_start_end_pos, d_pose]
                    else:
                        pose_state_info = ['NO_POSE_STATE', '']
                    d, prev_tense = motioncode_to_text(bp1, verb, prev_tense, mc[2][i_magg], bp2, bp1_initial, (mc[2][i_magg]['chronological_order']),
                                         pose_info=pose_state_info,
                                         simplified_captions=simplified_captions)



                    # choose the next value for bp1 (transition text)
                    # and_required = False
                    # if bp1 != " and ":
                    #     choices = [" and " + NO_VERB_KEY, special_trans, ", " + NO_VERB_KEY]
                    #     choices = [" and " + NO_VERB_KEY, special_trans, ", " + NO_VERB_KEY]
                    #     choices = [" XX " + NO_VERB_KEY, special_trans, ", " + NO_VERB_KEY]
                    #     if NO_VERB_KEY not in bp1: choices += [" and "] # Why?
                    #     bp1 = random.choice(choices)
                    # else:
                    #     bp1 = special_trans
                    # d = d.capitalize() if and_term == ''
                    # descriptions[win_ind] +=  inner_transition + d
                    # d_aggregated += inner_transition[i_magg] + d
                    d_aggregated += inner_transition + d

                    # checkpoint 1
                    and_required = random.choice([True, False])
                    if not and_required:
                        bp1 = random.choice(['' + NO_VERB_KEY, special_trans])
                        inner_transition = random.choice([', ', '. '])
                        if inner_transition == '. ':
                            bp1 = special_trans
                    else:
                        inner_transition  = ', and '
                        bp1 = random.choice(['' + NO_VERB_KEY, special_trans])

                    inner_transition = random.choice([', and ', '. ', ' and '])
                    # Adjustment based on the maximum allowed words after the
                    # last period to avoid long-sentences

                    word_count_since_period = len(d_aggregated[d_aggregated.rfind('.')+1:].split(' '))
                    if word_count_since_period >= SENTENCE_NAX_LENGTH:
                        inner_transition = '. '

                    if inner_transition == '. ':
                        bp1 = special_trans
                        prev_tense = 'None'
                    else:
                        bp1 = random.choice(['' + NO_VERB_KEY, special_trans])
                    # Adjustment based on the maximimum allowed words after the
                    # last time we mentioned the bp1 name (bp1_initial) to avoid
                    # ambiguity of long-range pronouns
                    if word_count_since_bp1initial > PRONOUN_MAX_WORDS:
                        bp1 = bp1_initial
                        # inner_transition = '. ' # ???
                        word_count_since_bp1initial = 0
                    else:
                        word_count_since_bp1initial += len(d.split(' '))

                    # Since and appears after timerelation
                    # bp1 = bp1.replace('and', '')

                    # Concatenation to the current description AND PREV STEP LAST ONE IF IT IS HAPPENING IN A NEW WINDOW
                    # TODO: implement time code and chronological codes and then apply all eligibility interpret and so forth
                    # descriptions[win_ind] += transitions[i_mc] + d
            if d_aggregated == '':
                print()

            # Adjustment
            # if word_count_since_bp1initial > PRONOUN_MAX_WORDS:
            #     transitions[i_mc] = '. '
            #     word_count_since_period = 0
            descriptions[win_ind] += transitions[i_mc] + d_aggregated

        # descriptions[win_ind] += f".<-(w={win_ind}) "  # end of the description

        descriptions[win_ind] += "." if descriptions[win_ind] != '' else ''

        # Fixing the spelling of verbs with tenses:
        # Iterate through spelling change patterns
        for miss_spell, replacement in SPELLING_CORRECTIONS:
            descriptions[win_ind] = descriptions[win_ind].replace(miss_spell, replacement)


        # Correct syntax (post-processing)
        # - removing wide spaces,
        # - replacing "upperarm" by "upper arm"
        # - randomly replacing all "their"/"them" by "his/him" or "her/her" depending on the chosen determiner
        # - capitalizing when beginning a sentence
        descriptions[win_ind] = re.sub("\s\s+", " ", descriptions[win_ind])
        descriptions[win_ind] = descriptions[win_ind].replace("upperarm", "upper arm")



        descriptions[win_ind] = descriptions[win_ind].replace(" .", ".").replace(' ,', ',')

        # descriptions[win_ind] = '. '.join(x.capitalize() for x in descriptions[win_ind].split('. '))
        # descriptions[win_ind] = '. '.join(x.strip().capitalize() for x in descriptions[win_ind].split('. '))

        # Captializing the "l" in the "l shape" and "L shape"
        descriptions[win_ind] = descriptions[win_ind].replace('l shape', 'L shape')
        descriptions[win_ind] = descriptions[win_ind].replace('l-shape', 'L shape')


        # Todo: this should go out of the for loop because
        #  the determiner should be consistent through all windows
        # Payam: fixing this by replacing determiner to global_determiner
        # determiner = global_determiner
        # if determiner in ["his", "her"]:
        #     # NOTE: do not replace "they" by "he/she" as "they" can sometimes
        #     # refer to eg. "the hands", "the feet" etc.
        #     # Extra-spaces allow to be sure to treat whole words only
        #     descriptions[win_ind] = descriptions[win_ind].replace(" their ", f" {determiner} ")
        #
        #     descriptions[win_ind] = descriptions[win_ind].replace("Their ", f"{determiner}")  # with the capital letter
        #     descriptions[win_ind] = descriptions[win_ind].replace(" them ", " him " if determiner == "his" else f" {determiner} ")





    # Final polish to have consistecy over determiners
    Gender = random.choice(['M', 'F'])
    selected_pronoun = "he" if Gender == 'M' else "she"
    selected_determiner = "his" if Gender == 'M' else "her"

    replacements = [selected_determiner, 'the']
    # Function to perform the replacement
    def random_replacer(match):
        return random.choice(replacements)
    for win_ind in range(nb_windows):
        descriptions[win_ind] = descriptions[win_ind].replace("<s/he>", selected_pronoun)

        # descriptions[win_ind] = descriptions[win_ind].replace(DETERMINER_KEY, selected_determiner)
        # In order to randomly pick one of them throughout the text
        descriptions[win_ind] = re.sub(re.escape(DETERMINER_KEY), random_replacer, descriptions[win_ind])

        descriptions[win_ind] = '. '.join(x.strip().capitalize() for x in descriptions[win_ind].split('. '))


    if verbose:
        print(f"Actual number of empty descriptions: {nb_actual_empty_description}.")

    return descriptions, determiners


################################################################################
## FORMAT BABEL INFORMATION
################################################################################

def sent_from_babel_tag(sents, need_ing=True):
    """Process a single tag at the time."""
    s = random.choice(sents)
    ss = s.split()
    ss_end = " "+" ".join(ss[1:]) 
    if len(ss) == 1:
        prefix = "" if need_ing else "is "
        s = prefix + random.choice([s, f"in a {s} action"])
    elif ss[0] == "do":
        if need_ing:
            s = "doing" + ss_end
        else:
            s = random.choice(["does", "is doing"]) + ss_end
    elif ss[0] == "having" and not need_ing:
        s = "has" + ss_end
    elif not need_ing:
        s = "is " + s
    return s

def create_sentence_from_babel_tags(pose_babel_tags, babel_tag2txt):
    pose_babel_text = []
    for pbt in pose_babel_tags:
        d = ""
        start_with_ing = random.random()>0.5
        prefix = " is " if start_with_ing else " "
        start = random.choice(SENTENCE_START) + prefix
        if len(pbt) > 1:
            tag1, tag2 = random.sample(pbt, 2)
            trans = random.choice([(" and ", False), (" while ", True)])
            d = start + sent_from_babel_tag(babel_tag2txt[tag1], start_with_ing) \
                + trans[0] + sent_from_babel_tag(babel_tag2txt[tag2], trans[1]) + ". "
        elif len(pbt) == 1:
            d = start + sent_from_babel_tag(babel_tag2txt[pbt[0]], start_with_ing) + ". "
        # small corrections
        if "They" in start: # made consistent with the chosen determiner in the main function
            d = d.replace(" is ", " are ")

        pose_babel_text.append(d)

    return pose_babel_text

def create_sentence_from_babel_to_hml3d(HML3D_BABEL, babel_tag2txt, motion_id, start_time, end_time, GPT_Template):

    sequence_labels = HML3D_BABEL[motion_id.replace('M', '')]['sequence_labels']
    frame_labels = HML3D_BABEL[motion_id.replace('M', '')]['frame_labels']


    if sequence_labels is None and frame_labels is None:
        return '', ''

    # get a record of tags with no sentence correspondence
    null_tags = set([tag for tag in babel_tag2txt if not babel_tag2txt[tag]])
    possibilities = []
    all_captions = []
    if sequence_labels is not None:
        for seq_label in sequence_labels:
            act_cat = seq_label['act_cat']
            if not isinstance(act_cat, list):
                act_cat = [act_cat]
            for ac in act_cat:
                if ac is not None and ac not in null_tags and ac in babel_tag2txt:
                    if ac not in possibilities:
                        possibilities.append(ac)
                    all_captions.append((ac, 0, 0))
    if frame_labels is not None:
        for f_label in frame_labels:
            #         TODO: Check time
            overlap = max(0, min(f_label['HML3D_end_t'], end_time) - max(f_label['HML3D_start_t'], start_time))
            percentage_covered = (overlap / (end_time - start_time)) * 100 if overlap > 0 else 0

            act_cat = f_label['act_cat']
            if not isinstance(act_cat, list):
                act_cat = [act_cat]
            for ac in act_cat:
                if ac is not None and ac not in null_tags and ac in babel_tag2txt:
                    if percentage_covered > 50:
                        if ac not in possibilities:
                            possibilities.append(ac)


                    all_captions.append((ac, f_label['HML3D_start_t'], f_label['HML3D_end_t']))

    # make BABEL caption details table:
    def print_actions_table(actions):
        result = ''
        # Determine the maximum length of the action category for alignment
        max_action_length = max(len(action) for action, _, _ in actions) + 1
        # Header
        result += (f"{'Action'.ljust(max_action_length)}| {'Start Frame'.ljust(12)}| {'End Frame'}\n")
        result += ("-" * (max_action_length + 26)) + "\n"  # Adjust based on the length of the header

        # Rows
        for action, start_time, end_time in actions:
            action_display = action.ljust(max_action_length)
            if start_time == 0 and end_time == 0:
                result += (f"{action_display}| {'Sequence'.ljust(12)}|")
                result += '\n'
            else:
                start_frame_display = str(int(start_time*20)).ljust(12)
                end_frame_display = str(int(end_time*20))
                result += (f"{action_display}| {start_frame_display}| {end_frame_display}")
                result += '\n'
        return result
    caption_details = print_actions_table(all_captions) if len(all_captions) else ''
    if len(possibilities)==0:
        return '', caption_details

    # Here we add GPT-3 caption generator.
    if GPT_Template=='GPT':
        with open('gpt3_annotations_BABEL4MotionScript.json', 'r') as f:
            gpt3_BABEL4MotionScript = json.load(f)

        label = random.choice(possibilities)
        gpt3_4texts_result = gpt3_BABEL4MotionScript[label] if label in gpt3_BABEL4MotionScript else ['']
        return random.choice(gpt3_4texts_result), caption_details

    d = ''
    # This is where we decided to use only one category
    possibilities = [random.choice(possibilities)]
    try:
        start_with_ing = random.random()>0.5
        prefix = " is " if start_with_ing else " "
        start = random.choice(SENTENCE_START) + prefix
        if len(possibilities) > 1:
            tag1, tag2 = random.sample(possibilities, 2)
            trans = random.choice([(" and ", False), (" while ", True)])
            d = start + sent_from_babel_tag(babel_tag2txt[tag1], start_with_ing) \
                + trans[0] + sent_from_babel_tag(babel_tag2txt[tag2], trans[1]) + ". "
        elif len(possibilities) == 1:
            d = start + sent_from_babel_tag(babel_tag2txt[possibilities[0]], start_with_ing) + ". "
        # small corrections
        if "They" in start: # made consistent with the chosen determiner in the main function
            d = d.replace(" is ", " are ")
    except:
        print()


    return d, caption_details


################################################################################
## EXECUTED PART
################################################################################

def motioncodes_sanity_check(motioncodes, time_bin_info):

    Motion_Bins = [[] for _ in range(time_bin_info['nb_binds'])]
    nb_motions = len(motioncodes)
    for m in range(nb_motions):
        bin_number = motioncodes[m][2]['start'] // time_bin_info['bin_size']
        Motion_Bins[bin_number].append(motioncodes[m])
    str_out, list_out = "", []
    for i, bin_motions in enumerate(Motion_Bins): # to skip redundant bins
        if len(Motion_Bins[i])==0 and i+1 < len(Motion_Bins):
            flag_empty_after = True
            for j in range(i+1, len(Motion_Bins)):
                if len(Motion_Bins[j])>0:
                    flag_empty_after = False
            if flag_empty_after:
                break
        str_out += (f'{"-" * 65}Bin number: {i}{"-" * 65}\n')
        str_out += (f"{'1st Joint':<30}{'Details':<81}{'2nd Joint':<30}\n")
        for j, motion in enumerate(bin_motions):
            joint1 = str(motion[0:2])
            joint2 = str(motion[3:])

            details = f'Start: {motion[2]["start"]:<4} End: {motion[2]["end"]:<4} ' \
                      f'spatial: {INTERPRETATION_SET_MOTION[motion[2]["spatial"]]:<25} ' \
                      f'Temporal: {INTERPRETATION_SET_MOTION[motion[2]["temporal"]] if motion[2]["temporal"] is not None else "" :<10}'

            str_out += (f"{joint1:<30}{str(details):<80}{joint2:<30}\n")
            str_out += ('_' * 113)
            str_out +=('\n\n')
            current_motioncode = {  'bin_number': i,
                                    'start': motion[2]["start"],
                                    'end': motion[2]["end"],
                                    'spatial': INTERPRETATION_SET_MOTION[motion[2]["spatial"]],
                                    'temporal': INTERPRETATION_SET_MOTION[motion[2]["temporal"]] if motion[2]["temporal"] is not None else "",
                                    'joint1': joint1,
                                    'joint2': joint2}

            list_out.append(current_motioncode)
            # print(str_out)

    print(str_out)
    return str_out, list_out



def motioncode_stat_analysis(coords, save_dir):
    # Input
    prop_eligible = 0.4
    prop_unskippable = 0.06

    # Prepare posecode queries
    # (hold all info about posecodes, essentially using ids)
    p_queries = prepare_posecode_queries()
    sp_queries = prepare_super_posecode_queries(p_queries)
    m_queries = prepare_motioncode_queries()

    # Infer posecodes
    saved_filepath = os.path.join(save_dir, "stat_motioncode_intptt_eligibility.pt")
    if os.path.isfile(saved_filepath) and False:
        p_interpretations, p_eligibility, INTPTT_NAME2IDX = torch.load(saved_filepath)
        print("Load file:", saved_filepath)
    else:
        # Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
        coords = prepare_input(coords)
        # Eval & interprete & elect eligible elementary posecodes
        p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries, verbose=True)

        m_interpretations, m_eligibility = infer_motioncodes(coords, p_interpretations, p_queries, sp_queries,
                                                             m_queries,
                                                             request='Low_Level_intptt',
                                                             verbose=True)
        # save
        torch.save([m_interpretations, m_eligibility, INTPTT_NAME2ID], saved_filepath)
        print("Saved file:", saved_filepath)

    # Get stats for super-posecodes
    # sp_params = [p_eligibility, sp_queries, prop_eligible, prop_unskippable]
    # superposecode_stats(*sp_params)

    # Get stats for elementary posecodes
    params = [m_interpretations, m_queries, None, None, "", prop_eligible, prop_unskippable]
    motioncode_intptt_scatter("angular", *params, save_fig="angle_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("proximity", *params, save_fig="dist_stats_1.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("spatial_relation_x", *params, save_fig="dist_stats_1.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("spatial_relation_y", *params, save_fig="dist_stats_1.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("spatial_relation_z", *params, save_fig="dist_stats_1.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("displacement_x", *params, save_fig="dist_stats_2.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("displacement_y", *params,  save_fig="dist_stats_3.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("displacement_z", *params, save_fig="posX_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("rotation_pitch", *params, save_fig="posY_stats_1.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("rotation_roll", *params, save_fig="posY_stats_2.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("rotation_yaw", *params, save_fig="posY_stats_3.pdf", save_dir=save_dir)

    # ADD_POSECODE_KIND

def motioncode_stat_analysis_step1_extraction(coords, save_dir):
    # Input
    prop_eligible = 0.4
    prop_unskippable = 0.06

    # Prepare posecode queries
    # (hold all info about posecodes, essentially using ids)
    p_queries = prepare_posecode_queries()
    sp_queries = prepare_super_posecode_queries(p_queries)
    m_queries = prepare_motioncode_queries()

    # Infer posecodes
    saved_filepath = os.path.join(save_dir, "stat_motioncode_intptt_eligibility.pt")
    if os.path.isfile(saved_filepath) and False:
        p_interpretations, p_eligibility, INTPTT_NAME2IDX = torch.load(saved_filepath)
        print("Load file:", saved_filepath)
    else:
        # Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
        coords = prepare_input(coords)
        # Eval & interprete & elect eligible elementary posecodes
        p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries, verbose=True)

        m_interpretations, m_eligibility = infer_motioncodes(coords, p_interpretations, p_queries, sp_queries,
                                                             m_queries,
                                                             request='Low_Level_intptt',
                                                             verbose=True)
        # save
        torch.save([m_interpretations, m_eligibility, INTPTT_NAME2ID], saved_filepath)
        print("Saved file:", saved_filepath)

    # Get stats for super-posecodes
    # sp_params = [p_eligibility, sp_queries, prop_eligible, prop_unskippable]
    # superposecode_stats(*sp_params)

    return m_interpretations

def motioncode_stat_analysis_step2_visualization(m_interpretations_all, save_dir):
    # calculate ration
    stats_text = "Ratio of motioncode types over all detected motioncodes:\n"
    kinds_total = dict()
    for m_kind in m_interpretations_all:
        kind_total = sum([len(joint_set) for joint_set in m_interpretations_all[m_kind]])
        kinds_total[m_kind] = kind_total
    total = sum(kinds_total[m_kind] for m_kind in kinds_total)
    for m_kind in kinds_total:
        stats_text += (
            # f"Ration {m_kind} ---> {kinds_total[m_kind]} / {total}    ||    {(kinds_total[m_kind] / total) * 100:.3}%\n")
            f"{m_kind:<{20}} ---> {kinds_total[m_kind]:>5} / {total:<5}    ||    {(kinds_total[m_kind] / total) * 100:>6.2f}%\n")

    with open(os.path.join(save_dir, "statistics.txt"), 'w') as fopen:
        fopen.write(stats_text)

    # Input
    prop_eligible = 0.4
    prop_unskippable = 0.06

    # Prepare posecode queries
    # (hold all info about posecodes, essentially using ids)
    p_queries = prepare_posecode_queries()
    sp_queries = prepare_super_posecode_queries(p_queries)
    m_queries = prepare_motioncode_queries()


    # Get stats for elementary posecodes
    params = [m_interpretations_all, m_queries, None, None, "", prop_eligible, prop_unskippable]
    motioncode_intptt_scatter("angular", *params, save_fig="angular_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("proximity", *params, save_fig="proximity_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("spatial_relation_x", *params, save_fig="spatial_X_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("spatial_relation_y", *params, save_fig="spatial_Y_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("spatial_relation_z", *params, save_fig="spatial_Z_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("displacement_x", *params, save_fig="disp_X_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("displacement_y", *params,  save_fig="disp_Y_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("displacement_z", *params, save_fig="disp_Z_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("rotation_pitch", *params, save_fig="rotation_pitch_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("rotation_roll", *params, save_fig="rotation_roll_stats.pdf", save_dir=save_dir)
    motioncode_intptt_scatter("rotation_yaw", *params, save_fig="rotation_yaw_stats.pdf", save_dir=save_dir)

    # ADD_POSECODE_KIND



if __name__ == "__main__" :

    import argparse
    from text2pose.config import POSESCRIPT_LOCATION

    parser = argparse.ArgumentParser(description='Parameters for captioning.')
    parser.add_argument('--action', default="generate_captions", choices=("generate_captions", "posecode_stats"), help="Action to perform.")
    parser.add_argument('--saving_dir', default=POSESCRIPT_LOCATION+"/generated_captions/", help='General location for saving generated captions and data related to them.')
    parser.add_argument('--version_name', default="tmp", help='Name of the caption version. Will be used to create a subdirectory of --saving_dir.')
    parser.add_argument('--simplified_captions', action='store_true', help='Produce a simplified version of the captions (basically: no aggregation, no omitting of some support keypoints for the sake of flow, no randomly referring to a body part by a substitute word).')
    parser.add_argument('--apply_transrel_ripple_effect', action='store_true', help='Discard some posecodes using ripple effect rules based on transitive relations between body parts.')
    parser.add_argument('--apply_stat_ripple_effect', action='store_true', help='Discard some posecodes using ripple effect rules based on statistically frequent pairs and triplets of posecodes.')
    parser.add_argument('--random_skip', action='store_true', help='Randomly skip some non-essential posecodes.')
    parser.add_argument('--add_babel_info', action='store_true', help='Add sentences using information extracted from BABEL.')
    parser.add_argument('--add_dancing_info', action='store_true', help='Add a sentence stating that the pose is a dancing pose if it comes from DanceDB, provided that --add_babel_info is also set to True.')

    args = parser.parse_args()

    # create saving location
    save_dir = os.path.join(args.saving_dir, args.version_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print("Created new dir", save_dir)

    # load and format joint coordinates for input (dict -> matrix)
    coords = torch.load(os.path.join(POSESCRIPT_LOCATION, "ids_2_coords_correct_orient_adapted.pt"))

    # </Payam> for test porpuse:
    keys_to_grab = ['0', '1', '2', '3']
    coords = {key: coords[key] for key in keys_to_grab if key in coords}

    pose_ids = sorted(coords.keys(), key=lambda k: int(k))
    coords = torch.stack([coords[k] for k in pose_ids])



    if args.action=="generate_captions":

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
                if pbt is None or pbt=="__BMLhandball__":
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
                    raise ValueError( str((i, pbt)) )

            # create a sentence from BABEL tags for each pose, if available
            pose_babel_text = create_sentence_from_babel_tags(pose_babel_tags, babel_tag2txt)

        # process
        t1 = time.time()
        main(coords,
                save_dir = save_dir,
                babel_info=pose_babel_text,
                simplified_captions=args.simplified_captions,
                apply_transrel_ripple_effect = args.apply_transrel_ripple_effect,
                apply_stat_ripple_effect = args.apply_stat_ripple_effect,
                random_skip = args.random_skip)
        print(f"Process took {time.time() - t1} seconds.")
        print(args)

    elif args.action == "posecode_stats":

        # Input
        prop_eligible = 0.4
        prop_unskippable = 0.06

        # Prepare posecode queries
        # (hold all info about posecodes, essentially using ids)
        p_queries = prepare_posecode_queries()
        sp_queries = prepare_super_posecode_queries(p_queries)

        # Infer posecodes
        saved_filepath = os.path.join(save_dir, "posecodes_intptt_eligibility.pt")
        if os.path.isfile(saved_filepath):
            p_interpretations, p_eligibility, INTPTT_NAME2ID = torch.load(saved_filepath)
            print("Load file:", saved_filepath)
        else:
            # Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
            coords = prepare_input(coords)
            # Eval & interprete & elect eligible elementary posecodes
            p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries, verbose=True)
            # save
            torch.save([p_interpretations, p_eligibility, INTPTT_NAME2ID], saved_filepath)
            print("Saved file:", saved_filepath)

        # Get stats for super-posecodes
        sp_params = [p_eligibility, sp_queries, prop_eligible, prop_unskippable]
        superposecode_stats(*sp_params)

        # Get stats for elementary posecodes
        params = [p_interpretations, p_queries, None, None, "", prop_eligible, prop_unskippable]
        posecode_intptt_scatter("angle", *params, save_fig="angle_stats.pdf", save_dir=save_dir)
        posecode_intptt_scatter("distance", *params, jx=0, jy=8, save_fig="dist_stats_1.pdf", save_dir=save_dir)
        posecode_intptt_scatter("distance", *params, jx=8, jy=14, save_fig="dist_stats_2.pdf", save_dir=save_dir)
        posecode_intptt_scatter("distance", *params, jx=14, jy=None, save_fig="dist_stats_3.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosX", *params, save_fig="posX_stats.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosY", *params, jx=0, jy=5, save_fig="posY_stats_1.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosY", *params, jx=5, jy=11, save_fig="posY_stats_2.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosY", *params, jx=11, jy=None, save_fig="posY_stats_3.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosZ", *params, jx=0, jy=5, save_fig="posZ_stats_1.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosZ", *params, jx=5, jy=None, save_fig="posZ_stats_2.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativeVAxis", *params, jy=6, save_fig="pitchroll_stats_1.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativeVAxis", *params, jx=6, save_fig="pitchroll_stats_2.pdf", save_dir=save_dir)
        posecode_intptt_scatter("onGround", *params, save_fig="ground_stats.pdf", save_dir=save_dir)
        # ADD_POSECODE_KIND