

# Captions are automatically generated based on the pieces of information
# contained in this file, regarding the different steps of the automatic
# captioning pipeline:
# - posecode extraction (eg. joint sets, interpretations)
# - posecode selection (eg. statistics-based rules to tackle redundancy)
# - posecode aggregation (eg. entities for entity-based aggregation rules)
# - posecode conversion (eg. template sentences)
# Note that some polishing actions are defined in captioning.py only.
# 
#
# General design choices:
# - use the 'left_hand'/'right_hand' virtual joints to represent the hands
# - use the 'torso' virtual joint to represent the torso (instead of eg. spine3)
# - the position of the hands wrt the rest of the body is very meaningful: let's
#     highlight all potential body connections with the distance posecodes.
#
#
# To define a new kind of posecodes, follow the ADD_POSECODE_KIND marks
# (also in posecodes.py)
# To define new super-posecodes,follow the ADD_SUPER_POSECODE marks
# To define new virtual joints, follow the ADD_VIRTUAL_JOINT marks
# (also in captioning.py)



################################################################################
#                                                                              #
#                   POSECODE EXTRACTION                                        #
#                                                                              #
################################################################################

######################
# POSECODE OPERATORS #
######################

# The following describes the different posecode operators (ie. kinds of relations):
# - category_names: names used for computation 
# - category_names_ticks: names used for display
# - category_thresholds: values (in degrees or meters) to distinguish between 2 categories
# - random_max_offset: value (in degrees or meters) used to randomize the binning step (noise level)

POSECODE_OPERATORS_VALUES = {
    'angle': { # values in degrees
        'category_names': ['completely bent', 'bent more',
                           'right angle', 'bent less',
                           'slightly bent', 'straight'],

        'category_names_ticks': ['completely bent', 'almost completely bent',
                                 'bent at right angle', 'partially bent',
                                 'slightly bent', 'straight'],

        'category_thresholds': [45, 75, 105, 135, 160],
        'random_max_offset': 5
    },
    'distance': { # values in meters
        'category_names': ['close', 'shoulder width', 'spread', 'wide'],
        'category_names_ticks': ['close', 'shoulder width', 'spread', 'wide'],
        'category_thresholds': [0.20, 0.50, 0.80],
        'random_max_offset': 0.05
    },

    'relativePosX': { # values in meters
        'category_names': ['at_right', 'ignored_relpos0', 'at_left'],
        'category_names_ticks': ['at the right of', 'x-ignored', 'at the left of'],
        'category_thresholds': [-0.15, 0.15],
        'random_max_offset': 0.05
    },
    'relativePosY': { # values in meters
        'category_names': ['below', 'ignored_relpos1', 'above'],
        'category_names_ticks': ['below', 'y-ignored', 'above'],
        'category_thresholds': [-0.15, 0.15],
        'random_max_offset': 0.05
    },
    'relativePosZ': { # values in meters
        'category_names': ['behind', 'ignored_relpos2', 'front'],
        'category_names_ticks': ['behind', 'z-ignored', 'in front of'],
        'category_thresholds': [-0.15, 0.15],
        'random_max_offset': 0.05
    },
    'relativeVAxis': { # values in degrees (between 0 and 90)
        'category_names': ['vertical', 'ignored_relVaxis', 'horizontal'],
        'category_names_ticks': ['vertical', 'pitch-roll-ignored', 'horizontal'],
        'category_thresholds': [10, 80],
        'random_max_offset': 5
    },
    'onGround': { # values in meters
        'category_names': ['on_ground', 'ignored_onGround'],
        'category_names_ticks': ['on ground', 'ground-ignored'],
        'category_thresholds': [0.10],
        'random_max_offset': 0.05
    },
    # ADD_POSECODE_KIND
    # ------------------------
    'position_x': { # values in meters
        'category_names': [f'x_{str(x / 100.0)}' for x in range(-200, 205, 5)] , # 81 bins considering 20*5cm bins per both sides
        # 'category_names': [f'x_{str(x / 100.0)}' for x in range(-200, 210, 10)] , # 41 bins considering 10*10cm bins per side and zero
        'category_names_ticks': [f'x_{str(x / 100.0)}' for x in range(-200, 205, 5)],
        'category_thresholds': [x / 100.0 for x in range(-200, 200, 5)],
        'random_max_offset': 0.05 #TODO: should decide about this too.
    },
    'position_y': { # values in meters
        'category_names': [f'x_{str(x / 100.0)}' for x in range(-200, 205, 5)] , # 81 bins considering 20*5cm bins per both sides
        'category_names_ticks': [f'x_{str(x / 100.0)}' for x in range(-200, 205, 5)],
        'category_thresholds': [x / 100.0 for x in range(-200, 200, 5)],
        'random_max_offset': 0.05
    },
    'position_z': { # values in meters
    'category_names': [f'x_{str(x / 100.0)}' for x in range(-200, 205, 5)] , # 81 bins considering 20*5cm bins per both sides
        'category_names_ticks': [f'x_{str(x / 100.0)}' for x in range(-200, 205, 5)],
        'category_thresholds': [x / 100.0 for x in range(-200, 200, 5)],
        'random_max_offset': 0.05
    },

    # ------------ORIENTATIONS------------
    # "leaning" would indicate a tilted position without necessarily
    # being parallel to the ground, while "lying" suggests a horizontal
    # and flat posture. For example, "leaning forward" implies a forward
    # tilt, while "lying flat" suggests being in a fully horizontal position.
    'orientation_pitch': { # values in degrees X-Axis  ***facing forward is 90***
        'category_names': ['upside_down_backward', 'lying_flat_backward', 'leaning_backward', 'slightly_leaning_backward',
                           'neutral_pitch',
                           'slightly_leaning_forward', 'leaning_forward', 'lying_flat_forward', 'upside_down_forwardt'
                           ],
        'category_names_ticks': ['upside_down_backward', 'lying_flat_backward', 'leaning_backward', 'slightly_leaning_backward',
                                 'neutral_pitch',
                                 'slightly_leaning_forward', 'leaning_forward', 'lying_flat_forward', 'upside_down_forwardt'],
        # 'category_thresholds': [ 20, 45, 75, 105, 135, 160],
        'category_thresholds': [35, 55, 75, 85, 95, 105, 125, 145],
        'random_max_offset': 5
    },
    'orientation_roll': { # values in degrees Y axis
        'category_names': ['upside_down_right', 'lying_right', 'leaning_right', 'moderately_leaning_right', 'slightly_leaning_right',
                           'neutral',
                           'slightly_leaning_left', 'moderately_leaning_left', 'leaning_left', 'lying_left', 'upside_down_left'],

        'category_names_ticks': ['upside down right', 'lying right', 'leaning right', 'moderately leaning right', 'slightly leaning right',
                                 'neutral',
                                 'slightly leaning left', 'moderately leaning left', 'leaning left', 'lying left', 'upside down left'],
        'category_thresholds': [-90, -45, -30, -15, 5, 5, 15, 30, 45, 90],
        'random_max_offset': 5
    },
    'orientation_yaw': { # values in degrees Z-Axis  ***facing forward is 0***
        'category_names': ['about-face_turned_clockwise', 'completely_turned_clockwise', 'moderately_turned_clockwise', 'slightly_turned_clockwise',
                           'neutral',
                           'slightly_turned_counterclockwise', 'moderately_turned_counterclockwise', 'completely_turned_counterclockwise', 'about-face_turned_counterclockwise'],

        'category_names_ticks': ['about-face_turned clockwise', 'completely turned clockwise', 'moderately turned clockwise', 'slightly turned clockwise',
                           'neutral',
                           'slightly turned counterclockwise', 'moderately turned counterclockwise', 'completely turned counterclockwise', 'about-face turned counterclockwise'],

        # 'category_thresholds': [-135, -90, -75, -20, 20, 75, 90, 135],
        'category_thresholds': [-135, -90, -60, -25, 25, 60, 90, 135],
        'random_max_offset': 5
    },


    #Todo:(XCV!)
    # posecode for translation like "significant_right" ???
    # Seems we do not need this thanks to the displacement motion code




}

MOTIONCODE_OPERATORS_VALUES = {
    'angular': { # values in degrees
        'name': 'Angular',

        'category_names': ['significant_bend', 'moderate_bend', 'slight_bend',
                           'no_action',
                           'slight_extension', 'moderate_extension', 'significant_extension'],

        'category_names_ticks': ['significant bend', 'moderate bend', 'slight bend',
                           'no action',
                           'slight extension', 'moderate extension', 'significant extension'],

        'category_thresholds': [ -4, -3, -2, 0, 2, 3], # Here we define thresholds like number of transitions


        'category_names_velocity': ['very_slow', 'slow', 'moderate', 'fast', 'very_fast'],
        'category_names_ticks_velocity': ['very slow', 'slow', 'moderate', 'fast', 'very fast'],
        'category_thresholds_velocity': [0.05, 0.1, 0.5, 0.8], # These thresholds apply to both extension and bend movements

        'random_max_offset': 1,
    },

    'proximity': {  # values in meters          ----Proximity modified due to lack of granularity at the distance posecode
        'name': 'Proximity',
        'category_names': ['significant_closing', 'moderate_closing', # 'slight_closing', #Neg. direction
                            'stationary',
                            # 'slight_spreading',
                            'moderate_spreading', 'significant_spreading' ],      # Pos direction
        'category_names_ticks': ['significant closing', 'moderate closing', # 'slight closing', #Neg. direction
                                    'stationary',
                                    #'slight spreading',
                                 'moderate spreading', 'significant spreading' ],
        'category_thresholds': [-2.1, -1, 1, 2.1], # [-4, -3, -2, 0, 2, 3],

        # Speed
        'category_names_velocity': ['very_slow', 'slow', 'moderate', 'fast', 'very_fast'],
        'category_names_ticks_velocity': ['very slow', 'slow', 'moderate', 'fast', 'very fast'],
        'category_thresholds_velocity': [0.05, 0.1, 0.5, 0.8], # These thresholds apply to both extension and bend movements

        'random_max_offset': 0.05,
    },

    'spatial_relation_x': {
        'name': 'SpatialRelation_X',
        # Categories for Spatial relational changes at the X-axis)
        'category_names': ['left-to-right', 'stationary', 'right-to-left'], # ['right-to-left', 'stationary', 'left-to-right'],
        'category_names_ticks': ['right-to-left', 'stationary', 'left-to-right'],
        'category_thresholds': [-1, +1],
        #[ ->, <-], Changed this due to left-to-right inconsistency observed
        # e.g. 000282 was fixed i.e.
        #       Plus fixing joint orientation around y-axis to get the first person view

        # velocity
        'category_names_velocity': ['very_slow', 'slow', 'moderate', 'fast', 'very_fast'],
        'category_names_ticks_velocity': ['very slow', 'slow', 'moderate', 'fast', 'very fast'],
        'category_thresholds_velocity': [0.05, 0.1, 0.5, 0.8],
        # These thresholds apply to both extension and bend movements

        'random_max_offset': 0.05,
    },
    'spatial_relation_y': {
        'name': 'SpatialRelation_Y',
        # Categories for Spatial relational changes at the Y-axis)
        'category_names': ['above-to-below', 'stationary', 'below-to-above'],
        'category_names_ticks': ['above-to-below', 'stationary', 'below-to-above'],
        'category_thresholds': [-1, 1],

        # velocity
        'category_names_velocity': ['very_slow', 'slow', 'moderate', 'fast', 'very_fast'],
        'category_names_ticks_velocity': ['very slow', 'slow', 'moderate', 'fast', 'very fast'],
        'category_thresholds_velocity': [0.05, 0.1, 0.5, 0.8],
        # These thresholds apply to both extension and bend movements

        'random_max_offset': 0.05,
    },
    'spatial_relation_z': {
        'name': 'SpatialRelation_Z',
        # Categories for Spatial relational changes at the Z-axis)
        'category_names': ['front-to-behind', 'stationary', 'behind-to-front'],
        'category_names_ticks': ['front-to-behind', 'stationary', 'behind-to-front'],
        'category_thresholds': [-1, 1],

        # velocity
        'category_names_velocity': ['very_slow', 'slow', 'moderate', 'fast', 'very_fast'],
        'category_names_ticks_velocity': ['very slow', 'slow', 'moderate', 'fast', 'very fast'],
        'category_thresholds_velocity': [0.05, 0.1, 0.5, 0.8],
        # These thresholds apply to both extension and bend movements

        'random_max_offset': 0.05,
    },

    'displacement_x': {
        'name': 'Displacement X',

        # Distance-based categories for leftward translation in the X-axis (negative X direction).
        # However, humuan tends to describe left/right translation based on his own egocenteric view,
        # not the animation character POV. Therefore we assume negative ...
        # 'category_names': ['very_long_right', 'long_right', 'moderate_right', 'short_right', 'very_short_right',
        #                             'no_action',
        #                             'very_short_left', 'short_left', 'moderate_left', 'long_left', 'very_long_left'],
        'category_names': ['very_long_left', 'long_left', 'moderate_left', 'short_left', 'very_short_left',
                                    'no_action',
                                    'very_short_right', 'short_right', 'moderate_right', 'long_right', 'very_long_right'],

        'category_names_ticks': ['very long left', 'long left', 'moderate left', 'short left', 'very short left',
                                    'no action',
                                    'very short right', 'short right', 'moderate right', 'long right', 'very long right'],

        'category_thresholds': [- 10, -8, -5, -3, -1, 1, 3, 5, 8, 10 ], # We assume each unit is 10cm, so our boundary would be [-2m, 2m] ??

        # Velocity
        'category_names_velocity': ['very_slow', 'slow', 'moderate', 'fast', 'very_fast'],
        'category_names_ticks_velocity': ['very slow', 'slow', 'moderate', 'fast', 'very fast'],
        'category_thresholds_velocity': [0.05, 0.1, 0.5, 0.8], # These thresholds apply to both extension and bend movements

        'random_max_offset': 0.05,

    },
    'displacement_y': {
    'name': 'Displacement Y',

    # Distance-based categories for upward translation in the Y-axis (positive Y direction)
    'category_names': ['very_long_down', 'long_down', 'moderate_down', 'short_down', 'very_short_down',
                                'no_action',
                                'very_short_up', 'short_up', 'moderate_up', 'long_up', 'very_long_up'],

    'category_names_ticks': ['very long down', 'long down', 'moderate down', 'short down', 'very short down',
                                'no action',
                                'very short up', 'short up', 'moderate up', 'long up', 'very long up'],

    'category_thresholds': [- 10, -8, -5, -3, -1, 1, 3, 5, 8, 10 ],  # We assume each unit is 10cm, so our boundary would be [-2m, 2m]

    # Velocity
    'category_names_velocity': ['very_slow', 'slow', 'moderate', 'fast', 'very_fast'],
    'category_names_ticks_velocity': ['very slow', 'slow', 'moderate', 'fast', 'very fast'],
    'category_thresholds_velocity': [0.05, 0.1, 0.5, 0.8],  # These thresholds apply to both extension and bend movements

    'random_max_offset': 0.05,
    },
    'displacement_z': {
        'name': 'Displacement Z',

        # Distance-based categories for forward translation in the Z-axis (positive Z direction)
        # Todo: not sure about the direction
        'category_names': ['very_long_backward', 'long_backward', 'moderate_backward', 'short_backward', 'very_short_backward',
                                    'no_action',
                                    'very_short_forward', 'short_forward', 'moderate_forward', 'long_forward', 'very_long_forward'],

        'category_names_ticks': ['very long forward', 'long forward', 'moderate forward', 'short forward', 'very short forward',
                                    'no action',
                                    'very short backward', 'short backward', 'moderate backward', 'long backward', 'very long backward'],

        'category_thresholds': [- 10, -8, -5, -3, -1, 1, 3, 5, 8, 10 ],  # We assume each unit is 10cm, so our boundary would be [-2m, 2m]

        # Velocity
        'category_names_velocity': ['very_slow', 'slow', 'moderate', 'fast', 'very_fast'],
        'category_names_ticks_velocity': ['very slow', 'slow', 'moderate', 'fast', 'very fast'],
        'category_thresholds_velocity': [0.05, 0.1, 0.5, 0.8],  # These thresholds apply to both extension and bend movements

        'random_max_offset': 0.05,
    },

    'rotation_pitch': {
        'name': 'Rotation_Pitch',
        # Categories for roll (rotation around X-axis)
        'category_names': ['significant_leaning_backward', 'moderate_leaning_backward', 'slight_leaning_backward',
                                'no_action',
                                'slight_leaning_forward', 'moderate_leaning_forward', 'significant_leaning_forward'],
        'category_names_ticks': ['significant_leaning_backward', 'moderate_leaning_backward', 'slight_leaning_backward',
                                'no_action',
                                'slight_leaning_forward', 'moderate_leaning_forward', 'significant_leaning_forward'],
        'category_thresholds': [-4, -3, -2, 0, 2, 3,],


        # Velocity
        'category_names_velocity': ['very_slow', 'slow', 'moderate', 'fast', 'very_fast'],
        'category_names_ticks_velocity': ['very slow', 'slow', 'moderate', 'fast', 'very fast'],
        'category_thresholds_velocity': [0.05, 0.1, 0.5, 0.8], # These thresholds apply to both extension and bend movements

        'random_max_offset': 0.05,


    },
    'rotation_roll': {
        'name': 'Rotation_Roll',
        # Categories for roll (rotation around Y-axis)
        'category_names': ['significant_leaning_right', 'moderate_leaning_right', 'slight_leaning_right',
                                'no_action',
                                'slight_leaning_left', 'moderate_leaning_left', 'significant_leaning_left'],
        'category_names_ticks': ['significant leaning right', 'moderate leaning right', 'slight leaning right',
                                'no action',
                                'slight leaning left', 'moderate leaning left', 'significant leaning left'],
        'category_thresholds': [-4, -3, -2, 0, 2, 3,],

        # Speed
        'category_names_velocity': ['very_slow', 'slow', 'moderate', 'fast', 'very_fast'],
        'category_names_ticks_velocity': ['very slow', 'slow', 'moderate', 'fast', 'very fast'],
        'category_thresholds_velocity': [0.05, 0.1, 0.5, 0.8], # These thresholds apply to both extension and bend movements

        'random_max_offset': 0.05,

    },
    'rotation_yaw': {
        'name': 'Rotation_Yaw',
        # Categories for yaw (rotation around Z-axis)
        'category_names': ['significant_turn_clockwise', 'moderate_turn_clockwise', 'slight_turn_clockwise',
                                'no_action',
                                'slight_turn_counterclockwise', 'moderate_turn_counterclockwise', 'significant_turn_counterclockwise'],
        'category_names_ticks': ['significant turn clockwise', 'moderate turn clockwise', 'slight turn clockwise',
                                'no action',
                                'slight turn counterclockwise', 'moderate turn counterclockwise', 'significant turn counterclockwise'],
        'category_thresholds': [-4, -3, -2, 0, 2, 3,],

        # velocity
        'category_names_velocity': ['very_slow', 'slow', 'moderate', 'fast', 'very_fast'],
        'category_names_ticks_velocity': ['very slow', 'slow', 'moderate', 'fast', 'very fast'],
        'category_thresholds_velocity': [0.05, 0.1, 0.5, 0.8], # These thresholds apply to both extension and bend movements

        'random_max_offset': 0.05,


    },



    #Todo (XCV!)
    # Seems we need to check if translation is a required for movement in the space?


    # Add other motioncodes if needed...
}

TIMECODE_OPERTATOR_VALUES = {
        'ChronologicalOrder': {  # values in fraction of seconds
        'name': 'ChronologicalOrder',
        'category_names': ["preceding_a_moment",    # This phrase indicates a bit more delay compared to
                                                    # "Soon before." The second action happened before a
                                                    # brief moment.
                            "soon_before",          # This implies that the second action happened soon,
                                                    # before but there may be a bit of a delay.
                            "shortly_before",       # This suggests that the second action occurred a short
                                                    # time before the first action, but not instantly.
                            "immediately_before",   # This indicates that the second action occurred right
                                                    # before the first action with no delay.

                            "simultaneously",        # This means both actions occur at the exact same time.

                           "immediately_after",     # This indicates that the second action occurs right
                                                    # after the first action with no delay.
                           "shortly_after",         # This suggests that the second action occurs a short
                                                    # time after the first action, but not instantly.
                           "soon_after",            # This implies that the second action happens soon,
                                                    # after but there may be a bit of a delay.
                           "after_a_moment"         # This phrase indicates a bit more delay compared to
                                                    # "Soon After." The second action happens after a
                                                    # brief moment.
                           ],
        'category_names_ticks': ["preceding_a_moment",
                                 "soon_before",
                                 "shortly_before",
                                 "immediately_before",

                                 "simultaneously",

                                 "immediately_after",
                                 "shortly_after",
                                 "soon_after",
                                 "after_a_moment"
                                 ],
        # 'category_thresholds': [-5, -3, -1.5, -.5, 0.5, 1.5, 3, 5], # in a fraction of seconds (regardless of fps)
        # 'category_thresholds': [-1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5], # in a fraction of seconds (regardless of fps)
        'category_thresholds': [-3, -1.7, -0.9, -0.2, 0.2, 0.9, 1.7, 3], # in a fraction of seconds (regardless of fps)

        'random_max_offset': 0.1,                   # in fraction of seconds
    },
}

########################
# ELEMENTARY POSECODES #
########################

# Next, we define the different posecodes to be studied for each kind of relation. 
# Descriptive structures are organized as follow:
# list, with sublists of size 5
# - joint set (joints involved in the computation of the posecode)
# - main body part (~ description topic) when converting the posecode to text.
#       If None, then the posecode can be used to describe either one of the
#       joint from the joint set (ie. any joint can be the description topic).
# - list of acceptable interpretations for description regarding the posecode
#       operator. If an empty list is provided, all interpretations from the
#       operator are to be considered (note that if no interpretation is to be
#       considered, then the posecode should not be defined in the first place).
# - list of rare interpretations, that should make it to the description
#       regardless of the random skip option. If an empty list is provided, it
#       means that there are no rare interpretations for the corresponding kind
#       of posecode and joint set. All rare interpretations must appear in the
#       list of acceptable interpretations.
# - list of 'support' interpretations, ie. posecode interpretations that are
#       used in intermediate computations to infer super-posecodes. There are 2
#       types of support interpretations, depending on what happens to them
#       after the super-posecode they contribute to is produced or not (a
#       super-posecode is not produced, for instance, when some of the
#       contributing posecodes do not have the required interpretation):
#           - type I ("support"): posecode interpretations that only exist to
#               help super-posecode inference, and will not make it to the
#               description text anyway. In other words, if the posecode's
#               interpretation is a support-I interpretation, then the posecode
#               interpretation becomes un-eligible for description after the
#               super-posecode inference step (ie. the support interpretation is
#               not an acceptable interpretation anymore).
#           - type II ("semi-support"; persistent): posecode interpretations
#               that will become un-eligible if the super-posecode is produced
#               (no matter how, provided that the support-II posecode
#               interpretation was the required one in some other possible
#               production recipe for the given super-posecode) and will remain
#               as acceptable interpretations otherwise.
#       Elements in this list must be formated as follow:
#         ==> (interpretation name (string), support type (int)).
#       It should be noted that:
#       * All support interpretations must appear in the list of acceptable
#           interpretations.
#       * Only support-II posecode interpretations can be rare interpretations.
#           Support-I posecode interpretations cannot be rare as they won't make
#           it to the description alone (it is the super-posecode to which they
#           contribute that determines whether they will "be" a rare
#           interpretation or not, by it being a rare production itself).
#       * An interpretation that is used to infer a super-posecode but is not a
#           support interpretation of any type will make it to the description
#           text, no matter if the super-posecode could be produced or not (this
#           is somewhat the opposite of a support-I interpretation).
#
# NOTE: this section contains information about posecode selection in the sense
# that rare and eligible posecode interpretations are defined here.


PLURAL_KEY = '<plural>' # use this key before a body topic (eg. feet/hands) if it is plural, as eg. f'{PLURAL_KEY}_feet'


#**********#
#  ANGLES  #
#**********#

ANGLE_POSECODES = [
    #*****************************************
    ### SEMANTIC: BENT JOINT?
    # L knee
    [('left_hip', 'left_knee', 'left_ankle'), 'left_knee',
        [], ['completely bent'], [('completely bent', 2)]],
    # R knee
    [('right_hip', 'right_knee', 'right_ankle'), 'right_knee',
        [], ['completely bent'], [('completely bent', 2)]],
    # L elbow
    [('left_shoulder', 'left_elbow', 'left_wrist'), 'left_elbow',
        [], ['completely bent'], []],
    # R elbow
    [('right_shoulder', 'right_elbow', 'right_wrist'), 'right_elbow',
        [], ['completely bent'], []]
]


#*************#
#  DISTANCES  #
#*************#

DISTANCE_POSECODES = [
    #*****************************************
    ### SEMANTIC: HOW CLOSE ARE SYMMETRIC BODY PARTS?
    [('left_elbow', 'right_elbow'), None, ["close", "shoulder width", "wide"], ["close"], [('shoulder width', 1)]], # elbows
    [('left_hand', 'right_hand'), None, ["close", "shoulder width", "spread", "wide"], [], [('shoulder width', 1)]], # hands
    [('left_knee', 'right_knee'), None, ["close", "shoulder width", "wide"], ["wide"], [('shoulder width', 1)]], # knees
    [('left_foot', 'right_foot'), None, ["close", "shoulder width", "wide"], ["close"], [('shoulder width', 1)]], # feet
    #*****************************************
    ### SEMANTIC: WHAT ARE THE HANDS CLOSE TO?
    [('left_hand', 'left_shoulder'), 'left_hand', ['close'], ['close'], []], # hand/shoulder... LL
    [('left_hand', 'right_shoulder'), 'left_hand', ['close'], ['close'], []], # ... LR
    [('right_hand', 'right_shoulder'), 'right_hand', ['close'], ['close'], []], # ... RR
    [('right_hand', 'left_shoulder'), 'right_hand', ['close'], ['close'], []], # ... RL
    [('left_hand', 'right_elbow'), 'left_hand', ['close'], ['close'], []], # hand/elbow LR (NOTE: LL & RR are impossible)
    [('right_hand', 'left_elbow'), 'right_hand', ['close'], ['close'], []], # ... RL
    [('left_hand', 'left_knee'), 'left_hand', ['close'], ['close'], []], # hand/knee... LL
    [('left_hand', 'right_knee'), 'left_hand', ['close'], ['close'], []], # ... LR
    [('right_hand', 'right_knee'), 'right_hand', ['close'], ['close'], []], # ... RR
    [('right_hand', 'left_knee'), 'right_hand', ['close'], ['close'], []], # ... RL
    [('left_hand', 'left_ankle'), 'left_hand', ['close'], ['close'], []], # hand/ankle... LL
    [('left_hand', 'right_ankle'), 'left_hand', ['close'], ['close'], []], # ... LR
    [('right_hand', 'right_ankle'), 'right_hand', ['close'], ['close'], []], # ... RR
    [('right_hand', 'left_ankle'), 'right_hand', ['close'], ['close'], []], # ... RL
    [('left_hand', 'left_foot'), 'left_hand', ['close'], ['close'], []], # hand/foot... LL
    [('left_hand', 'right_foot'), 'left_hand', ['close'], ['close'], []], # ... LR
    [('right_hand', 'right_foot'), 'right_hand', ['close'], ['close'], []], # ... RR
    [('right_hand', 'left_foot'), 'right_hand', ['close'], ['close'], []] # ... RL
]


#*********************#
#  RELATIVE POSITION  #
#*********************#

# Since the joint sets are shared accross X-, Y- and Z- relative positioning
# posecodes, all these posecodes are gathered below (with the interpretation
# sublists (acceptable, rare, support) being divided into 3 specific
# sub-sublists for the X-, Y-, Z-axis respectively)

RELATIVEPOS_POSECODES = [
    #*****************************************
    ### SEMANTIC: HOW ARE POSITIONED SYMMETRIC BODY PARTS RELATIVELY TO EACH OTHER?
    # shoulders
    [('left_shoulder', 'right_shoulder'), None,
        [None, ['below', 'above'], ['behind', 'front']],
        [[],[],[]], [[],[],[]]],
    # elbows
    [('left_elbow', 'right_elbow'), None,
        [None, ['below', 'above'], ['behind', 'front']],
        [[],[],[]], [[],[],[]]],
    # hands
    [('left_hand', 'right_hand'), None,
        [['at_right'], ['below', 'above'], ['behind', 'front']],
        [['at_right'],[],[]], [[],[],[]]],
    # knees
    [('left_knee', 'right_knee'), None,
        [None, ['below', 'above'], ['behind', 'front']],
        [[],[],[]], [[],[('above', 2)],[]]],
    # foots
    [('left_foot', 'right_foot'), None,
        [['at_right'], ['below', 'above'], ['behind', 'front']],
        [['at_right'],[],[]], [[],[],[]]],
    #*****************************************
    ### SEMANTIC: LEANING BODY? KNEELING BODY ?
    # leaning to side, forward/backward
    [('neck', 'pelvis'), 'body',
        [['at_right', 'at_left'], None, ['behind', 'front']],
        [[],[],[]],
        [[('at_right', 1), ('at_left', 1)],[],[('behind', 1), ('front', 1)]]], # support for 'bent forward/backward and to the sides'
    [('left_ankle', 'neck'), 'left_ankle',
        [None, ['below'], None],
        [[],[],[]],
        [[],[('below', 1)],[]]], # support for 'bent forward/backward'
    [('right_ankle', 'neck'), 'right_ankle',
        [None, ['below'], None],
        [[],[],[]],
        [[],[('below', 1)],[]]], # support for 'bent forward/backward'
    [('left_hip', 'left_knee'), 'left_hip',
        [None, ['above'], None],
        [[],[],[]],
        [[],[('above', 1)],[]]], # support for 'kneeling'
    [('right_hip', 'right_knee'), 'left_hip',
        [None, ['above'], None],
        [[],[],[]],
        [[],[('above', 1)],[]]], # support for 'kneeling'
    #*****************************************
    ### SEMANTIC: CROSSING ARMS/LEGS? EXTREMITIES BELOW/ABOVE USUAL (1/2)?
    ### (for crossing: compare the position of the body extremity wrt to the 
    ###  closest joint to the torso in the kinematic chain)
    # left_hand
    [('left_hand', 'left_shoulder'), 'left_hand',
        [['at_right'], ['above'], None], # removed 'below' based on stats
        [[],[],[]], [[],[],[]]],
    # right_hand
    [('right_hand', 'right_shoulder'), 'right_hand',
        [['at_left'], ['above'], None], # removed 'below' based on stats
        [[],[],[]], [[],[],[]]],
    # left_foot
    [('left_foot', 'left_hip'), 'left_foot',
        [['at_right'], ['above'], None],
        [['at_right'],['above'],[]], [[],[],[]]],
    # right_foot
    [('right_foot', 'right_hip'), 'right_foot',
        [['at_left'], ['above'], None],
        [['at_left'],['above'],[]], [[],[],[]]],
    #*****************************************
    ### SEMANTIC: EXTREMITIES BELOW/ABOVE USUAL (2/2)?
    # left_hand
    [('left_wrist', 'neck'), 'left_hand',
        [None, ['above'], None], # removed 'below' based on stats
        [[],[],[]], [[],[],[]]],
    # right_hand
    [('right_wrist', 'neck'), 'right_hand',
        [None, ['above'], None], # removed 'below' based on stats
        [[],[],[]], [[],[],[]]],
    # left_hand
    [('left_hand', 'left_hip'), 'left_hand',
        [None, ['below'], None], # removed 'above' based on stats
        [[],[],[]], [[],[],[]]],
    # right_hand
    [('right_hand', 'right_hip'), 'right_hand',
        [None, ['below'], None], # removed 'above' based on stats
        [[],[],[]], [[],[],[]]],
    #*****************************************
    ### SEMANTIC: EXTREMITIES IN THE FRONT //or// BACK?
    # left_hand
    [('left_hand', 'torso'), 'left_hand', 
        [None, None, ['behind']], # removed 'front' based on stats
        [[],[],[]], [[],[],[]]],
    # right_hand
    [('right_hand', 'torso'), 'right_hand', 
        [None, None, ['behind']], # removed 'front' based on stats
        [[],[],[]], [[],[],[]]],
    # left_foot
    [('left_foot', 'torso'), 'left_foot', 
        [None, None, ['behind', 'front']],
        [[],[],[]], [[],[],[]]],
    # right_foot
    [('right_foot', 'torso'), 'right_foot', 
        [None, None, ['behind', 'front']],
        [[],[],[]], [[],[],[]]],
]


#********************#
#  RELATIVE TO AXIS  #
#********************#

RELATIVEVAXIS_POSECODES = [
    #*****************************************
    ### SEMANTIC: BODY PART HORIZONTAL/VERTICAL?
    [('left_hip', 'left_knee'), 'left_thigh', ['horizontal', 'vertical'], ['horizontal'], []], # L thigh alignment
    [('right_hip', 'right_knee'), 'right_thigh', ['horizontal', 'vertical'], ['horizontal'], []], # R ...
    [('left_knee', 'left_ankle'), 'left_calf', ['horizontal', 'vertical'], ['horizontal'], []], # L calf alignment
    [('right_knee', 'right_ankle'), 'right_calf', ['horizontal', 'vertical'], ['horizontal'], []], # R ...
    [('left_shoulder', 'left_elbow'), 'left_upperarm', ['horizontal', 'vertical'], ['vertical'], []], # L upper arm alignment
    [('right_shoulder', 'right_elbow'), 'right_upperarm', ['horizontal', 'vertical'], ['vertical'], []], # R ...
    [('left_elbow', 'left_wrist'), 'left_forearm', ['horizontal', 'vertical'], ['vertical'], []], # L forearm alignment 
    [('right_elbow', 'right_wrist'), 'right_forearm', ['horizontal', 'vertical'], ['vertical'], []], # R ...
    [('pelvis', 'left_shoulder'), 'left_backdiag', ['horizontal'], [], [('horizontal', 1)]], # support for back/torso horizontality
    [('pelvis', 'right_shoulder'), 'right_backdiag', ['horizontal'], [], [('horizontal', 1)]], # support for back/torso horizontality
    [('pelvis', 'neck'), 'torso', ['vertical'], [], []], # back/torso alignment
    [('left_hand', 'right_hand'), f'{PLURAL_KEY}_hands', ['horizontal'], [], [('horizontal', 1)]],
    [('left_foot', 'right_foot'), f'{PLURAL_KEY}_feet', ['horizontal'], [], [('horizontal', 1)]],
]


#*************#
#  ON GROUND  #
#*************#

ONGROUND_POSECODES = [
    [('left_knee'), 'left_knee', ['on_ground'], [], [('on_ground', 1)]],
    [('right_knee'), 'right_knee', ['on_ground'], [], [('on_ground', 1)]],
    [('left_foot'), 'left_foot', ['on_ground'], [], [('on_ground', 1)]],
    [('right_foot'), 'right_foot', ['on_ground'], [], [('on_ground', 1)]],
]

#*************#
#  ON GROUND  #
#*************#



#*************#
#  POSITION  #
#*************#
POSITION_POSECODES_X = []
POSITION_POSECODES_Y = []
POSITION_POSECODES_Z = []
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
JOINT_NAMES = ALL_JOINT_NAMES[:22] + ALL_JOINT_NAMES[-2:] + ['left_middle2', 'right_middle2'] + VIRTUAL_JOINTS
#todo: this should be supports type I --> cuz we do not want them to be appeared in the final output
support_list_I_x = [(x, 1) for x in POSECODE_OPERATORS_VALUES['position_x']['category_names']]
support_list_I_y = [(x, 1) for x in POSECODE_OPERATORS_VALUES['position_y']['category_names']]
support_list_I_z = [(x, 1) for x in POSECODE_OPERATORS_VALUES['position_z']['category_names']]

support_list_II_x = [(x, 2) for x in POSECODE_OPERATORS_VALUES['position_x']['category_names']]
support_list_II_y = [(x, 2) for x in POSECODE_OPERATORS_VALUES['position_y']['category_names']]
support_list_II_z = [(x, 2) for x in POSECODE_OPERATORS_VALUES['position_z']['category_names']]

for joint in JOINT_NAMES:
    if joint == 'translation':
        POSITION_POSECODES_X.append([(joint,), 'body', [], [], support_list_I_x ])
        POSITION_POSECODES_Y.append([(joint,), 'body', [], [], support_list_I_y ])
        POSITION_POSECODES_Z.append([(joint,), 'body', [], [], support_list_I_z ])
    else:
        POSITION_POSECODES_X.append([(joint,), joint, [], [], support_list_I_x])
        POSITION_POSECODES_Y.append([(joint,), joint, [], [], support_list_I_y])
        POSITION_POSECODES_Z.append([(joint,), joint, [], [], support_list_I_z])


#*************#
#  POSITION  #
#*************#


'''
ORIENTATION_POSECODES = [
    #*****************************************
    ### SEMANTIC: ORIENTATION?
    [('orientation',), 'body',
        [], [], []],

    # We added this since the implementation requires a list rather than one item
    [('translation',), 'root_translation(Ignore)',
        [], [], []],
# Jim chat???
]
'''
#
ORIENTATION_PITCH_POSECODES = [
    #*****************************************
    ### SEMANTIC: ORIENTATION?
    [('orientation',), 'body',
        ['upside_down_forwardt', 'lying_flat_forward', 'leaning_forward',
         'leaning_backward', 'lying_flat_backward', 'upside_down_backward'],
        ['upside_down_forwardt', 'lying_flat_forward', 'leaning_forward',
         'leaning_backward', 'lying_flat_backward', 'upside_down_backward'],
        []
    ],
    # We added this since the implementation requires a list rather than one item
    [('translation',), 'root_translation(Ignore)',
        [], [], []],

]
ORIENTATION_ROLL_POSECODES = [
    #*****************************************
    ### SEMANTIC: ORIENTATION?
    [('orientation',), 'body',
        ['upside_down_right', 'lying_right', 'leaning_right', 'leaning_left', 'lying_left', 'upside_down_left'],
        ['upside_down_right', 'lying_right', 'lying_left', 'upside_down_left'],
        []
    ],
    # We added this since the implementation requires a list rather than one item
    [('translation',), 'root_translation(Ignore)',
        [], [], []],

]
ORIENTATION_YAW_POSECODES = [
    #*****************************************
    ### SEMANTIC: ORIENTATION?
    [('orientation',), 'body',
     ['about-face_turned_clockwise', 'completely_turned_clockwise', 'moderately_turned_clockwise',
      'moderately_turned_counterclockwise', 'completely_turned_counterclockwise', 'about-face_turned_counterclockwise'
     ],
     ['about-face_turned_clockwise', 'completely_turned_clockwise',
     'completely_turned_counterclockwise', 'about-face_turned_counterclockwise'],
        []
    ],
    # We added this since the implementation requires a list rather than one item
    [('translation',), 'root_translation(Ignore)',
        [], [], []],

]

# ADD_POSECODE_KIND (use a new '#***#' box, and define related posecodes below it)


##############################
## ALL ELEMENTARY POSECODES ##
##############################

ALL_ELEMENTARY_POSECODES = {
    "angle": ANGLE_POSECODES,
    "distance": DISTANCE_POSECODES,
    "relativePosX": [[p[0], p[1], p[2][0], p[3][0], p[4][0]] for p in RELATIVEPOS_POSECODES if p[2][0]],
    "relativePosY": [[p[0], p[1], p[2][1], p[3][1], p[4][1]] for p in RELATIVEPOS_POSECODES if p[2][1]],
    "relativePosZ": [[p[0], p[1], p[2][2], p[3][2], p[4][2]] for p in RELATIVEPOS_POSECODES if p[2][2]],
    "relativeVAxis": RELATIVEVAXIS_POSECODES,
    "onGround": ONGROUND_POSECODES,


    "position_x": POSITION_POSECODES_X,
    "position_y": POSITION_POSECODES_Y,
    "position_z": POSITION_POSECODES_Z,

    'orientation_pitch': ORIENTATION_PITCH_POSECODES,
    'orientation_roll': ORIENTATION_ROLL_POSECODES,
    'orientation_yaw': ORIENTATION_YAW_POSECODES

    # ADD_POSECODE_KIND
}

# kinds of posecodes for which the joints in the joint sets will *systematically*
# not be used for description (using the focus_body_part instead) 
POSECODE_KIND_FOCUS_JOINT_BASED = ['angle', 'relativeVAxis', 'onGround',

                                   "position_x",
                                   "position_y",
                                   "position_z",
                                   
                                   "orientation_pitch",
                                   "orientation_roll",
                                   "orientation_yaw",

                                   ] # ADD_POSECODE_KIND



# ***************************************************************************************************************
# Elementary Motion Codes

#**********#
#  ANGULAR  #
#**********#
acceptable_angular_temp = ['significant_bend', 'moderate_bend',
                           # 'slight_bend',
                           # 'no_action',
                           # 'slight_extension',
                           'moderate_extension', 'significant_extension']
ANGLULAR_MOTIONCODES = [
    #*****************************************
    ### SEMANTIC: BENT JOINT?
    # L knee
    [('left_hip', 'left_knee', 'left_ankle'), 'left_knee',
        # Spatial
        acceptable_angular_temp, # acceptable / eligibility
        ['significant_bend', 'significant_extension'], # Rare
        [('significant_bend', 2), ('significant_extension', 2)], # Supports
         # Temporal
         ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
         ['fast', 'very_fast'],  # Rare
         [('fast', 2), ('very_fast', 2)],  # Supports
     ],
    # R knee
    [('right_hip', 'right_knee', 'right_ankle'), 'right_knee',
        # Spatial
        acceptable_angular_temp,  # acceptable / eligibility
        ['significant_bend', 'significant_extension'],  # Rare
        [('significant_bend', 2), ('significant_extension', 2)],  # Supports
         # Temporal
         ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
         ['very_slow', 'fast', 'very_fast'],  # Rare
         [('fast', 2), ('very_fast', 2)],  # Supports
    ],

    # L elbow
    [('left_shoulder', 'left_elbow', 'left_wrist'), 'left_elbow',
        # Spatial
        acceptable_angular_temp,  # acceptable / eligibility
        ['significant_bend', 'significant_extension'],  # Rare
        [('significant_bend', 2), ('significant_extension', 2)],  # Supports
         # Temporal
         ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
         ['fast', 'very_fast'],  # Rare
         [('fast', 2), ('very_fast', 2)],  # Supports
    ],
    # R elbow
    [('right_shoulder', 'right_elbow', 'right_wrist'), 'right_elbow',
    # Spatial
        acceptable_angular_temp,  # acceptable / eligibility
        ['significant_bend', 'significant_extension'],  # Rare
        [('significant_bend', 2), ('significant_extension', 2)],  # Supports
         # Temporal
         ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
         ['fast', 'very_fast'],  # Rare
         [('fast', 2), ('very_fast', 2)],  # Supports
    ]
]

acceptable_proximity_temp = ['significant_closing', 'moderate_closing',
                             # 'slight_closing', #Neg. direction
                             # 'stationary',
                             # 'slight_spreading',
                             'moderate_spreading', 'significant_spreading' ]
acceptable_proximity_temp_Sig_only =  ['significant_closing', 'significant_spreading' ]
PROXIMITY_MOTIONCODES = [
    #*****************************************
    ### SEMANTIC: HOW CLOSE ARE SYMMETRIC BODY PARTS?           ---MOdified based on statistical analysis
    [('left_elbow', 'right_elbow'), None,
        # Spatial
        ["significant_closing", "moderate_closing", "moderate_spreading", "significant_spreading"],
        ["significant_closing", "significant_spreading"],
        [], # [], #[('moderate_closing', 1), ('moderate_spreading', 1)],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports

    ], # elbows

    [('left_hand', 'right_hand'), None,
            # Spatial
            ["significant_closing", "significant_spreading"],
            ["significant_closing", "significant_spreading"],
            [], # [('moderate_closing', 1), ('moderate_spreading', 1)],

            # Temporal
            ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
            ['fast', 'very_fast'],  # Rare
            [('fast', 2), ('very_fast', 2)],  # Supports
            ], # hands

    [('left_knee', 'right_knee'), None,
            # Spatial
            ["significant_closing", "significant_spreading"],
            ["significant_closing", "significant_spreading"],
            [], # [('moderate_closing', 1), ('moderate_spreading', 1)],
            # Temporal
            ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
            ['fast', 'very_fast'],  # Rare
            [('fast', 2), ('very_fast', 2)],  # Supports
    ], # knees


    [('left_foot', 'right_foot'), None,

        # Spatial
        ["significant_closing", "significant_spreading"],
        [], # ["significant_closing", "significant_spreading"],
        [], # [('moderate_closing', 1), ('moderate_spreading', 1)],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # feet

        # #********************XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX*********************
        # ### SEMANTIC: WHAT ARE THE HANDS CLOSE TO?
    [('left_hand', 'left_shoulder'), 'left_hand',
        # Spatial
        ["significant_closing", "moderate_closing", "moderate_spreading", "significant_spreading"],
        [],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # hand/shoulder... LL

    [('left_hand', 'right_shoulder'), 'left_hand',
         # Spatial
         ["significant_closing", "moderate_closing", "moderate_spreading", "significant_spreading"],
         ["significant_closing", "significant_spreading"],
         [],

         # Temporal
         ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
         ['fast', 'very_fast'],  # Rare
         [('fast', 2), ('very_fast', 2)],  # Supports
     ], # ... LR

    [('right_hand', 'right_shoulder'), 'right_hand',
        # Spatial
        ["significant_closing", "moderate_closing", "moderate_spreading", "significant_spreading"],
        ["significant_closing", "significant_spreading"],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # ... RR

    [('right_hand', 'left_shoulder'), 'right_hand',
        # Spatial
        ["significant_closing", "moderate_closing", "moderate_spreading", "significant_spreading"],
        ["significant_closing", "significant_spreading"],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # ... RL
    [('left_hand', 'right_elbow'), 'left_hand',
        # Spatial
        ["significant_closing", "moderate_closing", "moderate_spreading", "significant_spreading"],
        ["significant_closing", "significant_spreading"],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # hand/elbow LR (NOTE: LL & RR are impossible)

    [('right_hand', 'left_elbow'), 'right_hand',
         # Spatial
         ["significant_closing", "moderate_closing", "moderate_spreading", "significant_spreading"],
         ["significant_closing", "significant_spreading"],
         [],

         # Temporal
         ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
         ['fast', 'very_fast'],  # Rare
         [('fast', 2), ('very_fast', 2)],  # Supports
    ], # ... RL
    [('left_hand', 'left_knee'), 'left_hand',
        # Spatial
        ["significant_closing", "significant_spreading"],
        [],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # hand/knee... LL
    [('left_hand', 'right_knee'), 'left_hand',
        # Spatial
        ["significant_closing", "significant_spreading"],
        [],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # ... LR
    [('right_hand', 'right_knee'), 'right_hand',
        # Spatial
        ["significant_closing", "significant_spreading"],
        [],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # ... RR
    [('right_hand', 'left_knee'), 'right_hand',
        # Spatial
        ["significant_closing", "significant_spreading"],
        [],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # ... RL
    [('left_hand', 'left_ankle'), 'left_hand',
        # Spatial
        ["significant_closing","significant_spreading"],
        [],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # hand/ankle... LL
    [('left_hand', 'right_ankle'), 'left_hand',
        # Spatial
        ["significant_closing", "significant_spreading"],
        [],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # ... LR
    [('right_hand', 'right_ankle'), 'right_hand',
        # Spatial
        ["significant_closing", "significant_spreading"],
        [],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # ... RR
    [('right_hand', 'left_ankle'), 'right_hand',
        # Spatial
        ["significant_closing", "significant_spreading"],
        [],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # ... RL
    [('left_hand', 'left_foot'), 'left_hand',
        # Spatial
        ["significant_closing", "moderate_closing", "moderate_spreading", "significant_spreading"],
        ["significant_closing", "significant_spreading"],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # hand/foot... LL
    [('left_hand', 'right_foot'), 'left_hand',
        # Spatial
        ["significant_closing", "moderate_closing", "moderate_spreading", "significant_spreading"],
        ["significant_closing", "significant_spreading"],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # ... LR
    [('right_hand', 'right_foot'), 'right_hand',
        # Spatial
        ["significant_closing", "moderate_closing", "moderate_spreading", "significant_spreading"],
        ["significant_closing", "significant_spreading"],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ], # ... RR
    [('right_hand', 'left_foot'), 'right_hand',
        # Spatial
        ["significant_closing", "moderate_closing", "moderate_spreading", "significant_spreading"],
        ["significant_closing", "significant_spreading"],
        [],

        # Temporal
        ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
        ['fast', 'very_fast'],  # Rare
        [('fast', 2), ('very_fast', 2)],  # Supports
    ] # ... RL
]

ROTATION_MOTIONCODES = [
    #*****************************************
    [('orientation'), 'body', # 'Rotation Pitch'
     # Spatial
     [],  # acceptable / eligibility
     [  'significant_leaning_forward', 'moderate_leaning_forward',
        'moderate_leaning_backward', 'significant_leaning_backward'
     ],  # Rare
     [('significant_leaning_forward', 2), ('significant_leaning_backward', 2)],  # Supports
     # Temporal
     ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
     ['fast', 'very_fast'],  # Rare
     [('fast', 2), ('very_fast', 2)],  # Supports
     ],

    [('orientation'), 'body', # Rotation Roll
     # Spatial
     [],  # acceptable / eligibility
     ['significant_leaning_right', 'moderate_leaning_right',
      'moderate_leaning_left', 'significant_leaning_left'
      ],  # Rare
     [('significant_leaning_right', 2), ('significant_leaning_left', 2)],  # Supports
     # Temporal
     ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
     ['fast', 'very_fast'],  # Rare
     [('fast', 2), ('very_fast', 2)],  # Supports
     ],

    [('orientation'), 'body', # Rotation Yaw
     # Spatial
     [],  # acceptable / eligibility
     ['significant_turn_clockwise', 'moderate_turn_clockwise',
      'moderate_turn_counterclockwise', 'significant_turn_counterclockwise'
      ],  # Rare
     [('significant_turn_clockwise', 2), ('significant_turn_counterclockwise', 2)],  # Supports
     # Temporal
     ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
     ['fast', 'very_fast'],  # Rare
     [('fast', 2), ('very_fast', 2)],  # Supports
     ],

]

# Todo: this should be seperated for x y and z axes
SPATIAL_RELATION_X_MOTIONCODES = [

    #*****************************************
    ### SEMANTIC: HOW ARE MOVING SYMMETRIC BODY PARTS RELATIVELY TO EACH OTHER?
    # shoulders     No Need
    # elbows        No Need
    # hands
    [('left_hand', 'right_hand'), None,
        # Spatial
        [],  # acceptable / eligibility
        ['right-to-left', 'left-to-right'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # knees     No Need
    # Feet      No Need due to stats
    # [('left_foot', 'right_foot'), None,
    #     # Spatial
    #     [],  # acceptable / eligibility
    #     ['right-to-left', 'left-to-right'], # Rare
    #     [], # Supports
    #     # Temporal
    #     None,  # acceptable / eligibility
    #     [],  # Rare
    #     [],  # Supports
    # ],
    # *****************************************
    ### SEMANTIC: CROSSING ARMS/LEGS? EXTREMITIES BELOW/ABOVE USUAL (1/2)?
    ### (for crossing: compare the position of the body extremity wrt to the
    ###  closest joint to the torso in the kinematic chain)
    # left_hand
    [('left_hand', 'left_shoulder'), 'left_hand',
        # Spatial
        [],  # acceptable / eligibility
        [], #['right-to-left', 'left-to-right'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # right_hand
    [('right_hand', 'right_shoulder'), 'right_hand',
        # Spatial
        [],  # acceptable / eligibility
        [],  # ['right-to-left', 'left-to-right'],  # Rare
        [],  # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
     ],
    # left_foot
    [('left_foot', 'left_hip'), 'left_foot',
        # Spatial
        [],  # acceptable / eligibility
        [], # ['right-to-left', 'left-to-right'],  # Rare
        [],  # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
     ],
    # right_foot
    [('right_foot', 'right_hip'), 'right_foot',
        # Spatial
        [],  # acceptable / eligibility
        [], #['right-to-left', 'left-to-right'],  # Rare
        [],  # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
     ],

    #*****************************************
    ### SEMANTIC: EXTREMITIES BELOW/ABOVE USUAL (2/2)?
    # left_hand     No Need
    # right_hand    No Need
    # left_hand     No Need
    # right_hand    No Need

]
SPATIAL_RELATION_Y_MOTIONCODES = [

    #*****************************************
    ### SEMANTIC: HOW ARE MOVING SYMMETRIC BODY PARTS RELATIVELY TO EACH OTHER?
    # shoulders
    [('left_shoulder', 'right_shoulder'), None,
        # Spatial
        [],  # acceptable / eligibility
        [], # ['above-to-below', 'below-to-above'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # elbows
    [('left_elbow', 'right_elbow'), None,
        # Spatial
        [],  # acceptable / eligibility
        [], # ['above-to-below', 'below-to-above'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # hands
    [('left_hand', 'right_hand'), None,
        # Spatial
        [],  # acceptable / eligibility
        [], # ['above-to-below', 'below-to-above'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # knees
    [('left_knee', 'right_knee'), None,
        # Spatial
        [],  # acceptable / eligibility
        [], # ['above-to-below', 'below-to-above'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # foots
    [('left_foot', 'right_foot'), None,
        # Spatial
        [],  # acceptable / eligibility
        [], # ['above-to-below', 'below-to-above'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # *****************************************
    ### SEMANTIC: CROSSING ARMS/LEGS? EXTREMITIES BELOW/ABOVE USUAL (1/2)?
    ### (for crossing: compare the position of the body extremity wrt to the
    ###  closest joint to the torso in the kinematic chain)
    # left_hand
    [('left_hand', 'left_shoulder'), 'left_hand',
        # Spatial
        [],  # acceptable / eligibility
        [], # ['above-to-below', 'below-to-above'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # right_hand
    [('right_hand', 'right_shoulder'), 'right_hand',
        # Spatial
        [],  # acceptable / eligibility
        [], # ['above-to-below', 'below-to-above'],  # Rare
        [],  # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
     ],
    #
    [('left_foot', 'left_hip'), 'left_foot',
        # Spatial
        [],  # acceptable / eligibility
        ['above-to-below', 'below-to-above'],  # Rare
        [],  # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
     ],
    # right_foot
    [('right_foot', 'right_hip'), 'right_foot',
        # Spatial
        [],  # acceptable / eligibility
        ['above-to-below', 'below-to-above'],  # Rare
        [],  # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
     ],

    #*****************************************
    ### SEMANTIC: EXTREMITIES BELOW/ABOVE USUAL (2/2)?
    # left_hand
    [('left_wrist', 'neck'), 'left_hand',
        # Spatial
        [],  # acceptable / eligibility
        [], # ['above-to-below', 'below-to-above'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],

    # right_hand
    [('right_wrist', 'neck'), 'right_hand',
        # Spatial
        [],  # acceptable / eligibility
        [], # ['above-to-below', 'below-to-above'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],

    # left_hand
    [('left_hand', 'left_hip'), 'left_hand',
        # Spatial
        [],  # acceptable / eligibility
        [], # ['above-to-below', 'below-to-above'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # right_hand
    [('right_hand', 'right_hip'), 'right_hand',
        # Spatial
        [],  # acceptable / eligibility
        [], # ['above-to-below', 'below-to-above'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],

]
SPATIAL_RELATION_Z_MOTIONCODES = [

    #***************************************** 'front-to-behind', 'stationary', 'behind-to-front'
    ### SEMANTIC: HOW ARE MOVING SYMMETRIC BODY PARTS RELATIVELY TO EACH OTHER?
    # shoulders
    [('left_shoulder', 'right_shoulder'), None,
        # Spatial
        [],  # acceptable / eligibility
        [], # ['front-to-behind', 'behind-to-front'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # elbows
        [('left_elbow', 'right_elbow'), None,
        # Spatial
        [],  # acceptable / eligibility
        [], # ['front-to-behind', 'behind-to-front'],  # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # hands
    [('left_hand', 'right_hand'), None,
        # Spatial
        [],  # acceptable / eligibility
        [], # ['front-to-behind', 'behind-to-front'], # Rare
        [], # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
    ],
    # knees         #######---REMOVED---####### Due to stat analysis ??
    # [('left_knee', 'right_knee'), None,
    #     # Spatial
    #     [],  # acceptable / eligibility
    #     [], # ['front-to-behind', 'behind-to-front'], # Rare
    #     [], # Supports
    #     # Temporal
    #     None,  # acceptable / eligibility
    #     [],  # Rare
    #     [],  # Supports
    # ],
    # foots         #######---REMOVED---####### Due to stat analysis ??
    # [('left_foot', 'right_foot'), None,
    #     # Spatial
    #     [],  # acceptable / eligibility
    #     [], # ['front-to-behind', 'behind-to-front'], # Rare
    #     [], # Supports
    #     # Temporal
    #     None,  # acceptable / eligibility
    #     [],  # Rare
    #     [],  # Supports
    # ],
    # *****************************************
    ### SEMANTIC: CROSSING ARMS/LEGS? EXTREMITIES BELOW/ABOVE USUAL (1/2)?
    ### (for crossing: compare the position of the body extremity wrt to the
    ###  closest joint to the torso in the kinematic chain)
    # left_hand
    # right_hand
    # left_foot
    # right_foot

    #*****************************************
    ### SEMANTIC: EXTREMITIES BELOW/ABOVE USUAL (2/2)?
    # left_hand neck
    # right_hand necl
    # left_hand l_hip
    # right_hand r_hip
    #*****************************************
    ### SEMANTIC: EXTREMITIES IN THE FRONT //or// BACK?
    # left_hand
    [('left_hand', 'torso'), 'left_hand',
        # Spatial
        [],  # acceptable / eligibility
        [], # ['front-to-behind', 'behind-to-front'],  # Rare
        [],  # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
     ],
    # right_hand
    [('right_hand', 'torso'), 'right_hand',
        # Spatial
        [],  # acceptable / eligibility
        [], # ['front-to-behind', 'behind-to-front'],  # Rare
        [],  # Supports
        # Temporal
        None,  # acceptable / eligibility
        [],  # Rare
        [],  # Supports
     ],
    # # left_foot           #######---REMOVED---####### Due to stat analysis ??
    # [('left_foot', 'torso'), 'left_foot',
    #  # Spatial
    #  [],  # acceptable / eligibility
    #  [], # ['front-to-behind', 'behind-to-front'],  # Rare
    #  [],  # Supports
    #  # Temporal
    #  None,  # acceptable / eligibility
    #  [],  # Rare
    #  [],  # Supports
    #  ],
    # # right_foot          #######---REMOVED---####### Due to stat analysis ??
    # [('right_foot', 'torso'), 'right_foot',
    #  # Spatial
    #  [],  # acceptable / eligibility
    #  [], # ['front-to-behind', 'behind-to-front'],  # Rare
    #  [],  # Supports
    #  # Temporal
    #  None,  # acceptable / eligibility
    #  [],  # Rare
    #  [],  # Supports
    #  ],
]




# We only consider the tranlsation for this step because other joints
# are not being described using the displacement motioncode
# DISPLACEMENT_NOTIONCODES = []
# support_list_m_x = [(x, 1) for x in MOTIONCODE_OPERATORS_VALUES['displacement_x']['category_names']]
# support_list_m_y = [(x, 1) for x in MOTIONCODE_OPERATORS_VALUES['displacement_y']['category_names']]
# support_list_m_z = [(x, 1) for x in MOTIONCODE_OPERATORS_VALUES['displacement_z']['category_names']]
# # We set focus body part to None to skip further implication at the parse_posecode_joints in captioning.py
# for joint in JOINT_NAMES:
#     if joint == 'translation':
#         DISPLACEMENT_NOTIONCODES.append([(joint,), None, [], [], [], [], [], [] ]) #todo: this should not be supports type I
#     else:
#         DISPLACEMENT_NOTIONCODES.append([(joint,), None, [], [], support_list,     [], ['fast', 'very_fast'], []])  # todo: this should be supports type I

acceptable_motion_X = MOTIONCODE_OPERATORS_VALUES['displacement_x']['category_names'][:3] + \
                      MOTIONCODE_OPERATORS_VALUES['displacement_x']['category_names'][-3:]
acceptable_motion_Y = MOTIONCODE_OPERATORS_VALUES['displacement_y']['category_names'][:3] + \
                      MOTIONCODE_OPERATORS_VALUES['displacement_y']['category_names'][-3:]
acceptable_motion_Z = MOTIONCODE_OPERATORS_VALUES['displacement_z']['category_names'][:3] + \
                      MOTIONCODE_OPERATORS_VALUES['displacement_z']['category_names'][-3:]
rare_motion_X = MOTIONCODE_OPERATORS_VALUES['displacement_x']['category_names'][:2] + \
                      MOTIONCODE_OPERATORS_VALUES['displacement_x']['category_names'][-2:]
rare_motion_Y = MOTIONCODE_OPERATORS_VALUES['displacement_y']['category_names'][:2] + \
                      MOTIONCODE_OPERATORS_VALUES['displacement_y']['category_names'][-2:]
rare_motion_Z = MOTIONCODE_OPERATORS_VALUES['displacement_z']['category_names'][:2] + \
                      MOTIONCODE_OPERATORS_VALUES['displacement_z']['category_names'][-2:]
joint = 'translation'
DISPLACEMENT_NOTIONCODES_X = [ [(joint,), 'body',
                                 acceptable_motion_X, rare_motion_X, [],
                                 # Temporal
                                 ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
                                 ['fast', 'very_fast'],  # Rare
                                 [('fast', 2), ('very_fast', 2)],  # Supports
                                 ]
                             ]
DISPLACEMENT_NOTIONCODES_Y = [ [(joint,), 'body',
                                 acceptable_motion_Y, rare_motion_Y, [],
                                 # Temporal
                                 ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
                                 ['fast', 'very_fast'],  # Rare
                                 [('fast', 2), ('very_fast', 2)],  # Supports
                                 ]
                             ]
DISPLACEMENT_NOTIONCODES_Z = [ [(joint,), 'body',
                                 acceptable_motion_Z, rare_motion_Z, [],
                                 # Temporal
                                 ['moderate', 'fast', 'very_fast'],  # acceptable / eligibility
                                 ['fast', 'very_fast'],  # Rare
                                 [('fast', 2), ('very_fast', 2)],  # Supports
                                 ]
                             ]
# ----------------------------------------------------------------------------------------------------------------



##############################
## ALL ELEMENTARY MOTIONCODES ##
##############################

ALL_ELEMENTARY_MOTIONCODES = {
    "angular": ANGLULAR_MOTIONCODES,
    "proximity": PROXIMITY_MOTIONCODES,

    "spatial_relation_x": SPATIAL_RELATION_X_MOTIONCODES,
    "spatial_relation_y": SPATIAL_RELATION_Y_MOTIONCODES,
    "spatial_relation_z": SPATIAL_RELATION_Z_MOTIONCODES,

    'displacement_x': DISPLACEMENT_NOTIONCODES_X,
    'displacement_y': DISPLACEMENT_NOTIONCODES_Y,
    'displacement_z': DISPLACEMENT_NOTIONCODES_Z,

    'rotation_pitch': [ROTATION_MOTIONCODES[0]],
    'rotation_roll':  [ROTATION_MOTIONCODES[1]],
    'rotation_yaw':   [ROTATION_MOTIONCODES[2]]

    # # ADD_POSECODE_KIND
}

# kinds of posecodes for which the joints in the joint sets will *systematically*
# not be used for description (using the focus_body_part instead)
MOTIONCODE_KIND_FOCUS_JOINT_BASED = [   'angular',
                                        'displacement_x',
                                        'displacement_y',
                                        'displacement_z',

                                        'rotation_pitch',
                                        'rotation_roll',
                                        'rotation_yaw'

                                    #     We do not add spatial relation since it has a mixed set of
                                    #     queries with/without body part focus.
                                    ]
                                   # 'relativeVAxis', 'onGround',
                                   # "PositionX",                   TODO: We may add other motioncodes here
                                   # "PositionY",
                                   # "PositionZ"] # ADD_POSECODE_KIND

# ***************************************************************************************************************


# ***************************************************************************************************************
# ********************************************** TIMECODES *******************************************************
# ***************************************************************************************************************

GENERAL_TIMECODES = [
    #*****************************************
    ### SEMANTIC: All body part or joint appeared in the motion pool?
    # L knee
    [None, None, # Joint set and body part are None which means all.
        # Chronological order
        [], # acceptable / eligibility
        ["preceding_a_moment", "after_a_moment"], # Rare
        [], # Supports
     ],

]





#####################
## SUPER-POSECODES ##
#####################

# Super-posecodes are a specific kind of posecodes, defined on top of other
# ("elementary") posecodes. They can be seen as a form of (non-necessarily
# destructive) specific aggregation, that must happen before the posecode
# selection process. "Non- necessarily destructive" because only posecodes with
# support-I or support-II interpretation may be removed during super-posecode
# inference. Some super-posecodes can be produced using several different sets
# of elementary posecodes (hence the list+dict organization below). While they
# are built on top of elementary posecodes which have several possible
# (exclusive) interpretations, super-posecodes are not assumed to be like that
# (they are binary: either they could be produced or they could not). Hence, the
# interpretation matrix does not make much sense for super-posecodes: it all
# boils down to the eligibility matrix, indicating whether the posecode exists
# and is eligible for description.

# Organization:
# 1 list + 1 dict
# - list: super-posecodes definition
#       - super-posecode ID
#       - the super-posecode itself, ie. the joint set (names of the involved
#           body parts; or of the focus body part) + the interpretation
#       - a boolean indicating whether this is a rare posecode.
#           NOTE: super-posecodes are assumed to be always eligible for
#           description (otherwise, no need to define them in the first place).
# - dict: elementary posecode requirements to produce the super-posecodes
#       - key: super-posecode ID
#       - value: list of the different ways to produce the super-posecode, where
#           a way is represented by the list of posecodes required to produce
#           the super-posecode (posecode kind, joint set tuple (with joints in
#           the same order as defined for the posecode operator), required
#           interpretation). Required posecode interpretation are not necessarily
#           support-I or support-II interpretations.


SUPER_POSECODES = [
    ['torso_horizontal', [('torso'), 'horizontal'], True],
    ['body_bent_left', [('body'), 'bent_left'], False],
    ['body_bent_right', [('body'), 'bent_right'], False],
    ['body_bent_backward', [('body'), 'bent_backward'], True],
    ['body_bent_forward', [('body'), 'bent_forward'], False],
    ['kneel_on_left', [('body'), 'kneel_on_left'], True],
    ['kneel_on_right', [('body'), 'kneel_on_right'], True],
    ['kneeling', [('body'), 'kneeling'], True],
    ['hands_shoulder_width', [(f'{PLURAL_KEY}_hands'), 'shoulder width'], True],
    ['feet_shoulder_width', [(f'{PLURAL_KEY}_feet'), 'shoulder width'], False],
    # ADD_SUPER_POSECODE
]

SUPER_POSECODES_REQUIREMENTS = {
    'torso_horizontal': [
        [['relativeVAxis', ('pelvis', 'left_shoulder'), 'horizontal'],
         ['relativeVAxis', ('pelvis', 'right_shoulder'), 'horizontal']]],
    'body_bent_left': [
        # (way 1) using the left ankle
        [['relativePosY', ('left_ankle', 'neck'), 'below'],
         ['relativePosX', ('neck', 'pelvis'), 'at_left']],
        # (way 2) using the right ankle
        [['relativePosY', ('right_ankle', 'neck'), 'below'],
         ['relativePosX', ('neck', 'pelvis'), 'at_left']]],
    'body_bent_right': [
        # (way 1) using the left ankle
        [['relativePosY', ('left_ankle', 'neck'), 'below'],
         ['relativePosX', ('neck', 'pelvis'), 'at_right']],
        # (way 2) using the right ankle
        [['relativePosY', ('right_ankle', 'neck'), 'below'],
         ['relativePosX', ('neck', 'pelvis'), 'at_right']]],
    'body_bent_backward': [
        # (way 1) using the left ankle
        [['relativePosY', ('left_ankle', 'neck'), 'below'],
         ['relativePosZ', ('neck', 'pelvis'), 'behind']],
        # (way 2) using the right ankle
        [['relativePosY', ('right_ankle', 'neck'), 'below'],
         ['relativePosZ', ('neck', 'pelvis'), 'behind']]],
    'body_bent_forward': [
        # (way 1) using the left ankle
        [['relativePosY', ('left_ankle', 'neck'), 'below'],
         ['relativePosZ', ('neck', 'pelvis'), 'front']],
        # (way 2) using the right ankle
        [['relativePosY', ('right_ankle', 'neck'), 'below'],
         ['relativePosZ', ('neck', 'pelvis'), 'front']]],
    'kneel_on_left': [
        [['relativePosY', ('left_knee', 'right_knee'), 'below'],
         ['onGround', ('left_knee'), 'on_ground'],
         ['onGround', ('right_foot'), 'on_ground']]],
    'kneel_on_right': [
        [['relativePosY', ('left_knee', 'right_knee'), 'above'],
         ['onGround', ('right_knee'), 'on_ground'],
         ['onGround', ('left_foot'), 'on_ground']]],
    'kneeling': [
        # (way 1)
        [['relativePosY', ('left_hip', 'left_knee'), 'above'],
         ['relativePosY', ('right_hip', 'right_knee'), 'above'],
         ['onGround', ('left_knee'), 'on_ground'],
         ['onGround', ('right_knee'), 'on_ground']],
        # (way 2)
        [['angle', ('left_hip', 'left_knee', 'left_ankle'), 'completely bent'],
         ['angle', ('right_hip', 'right_knee', 'right_ankle'), 'completely bent'],
         ['onGround', ('left_knee'), 'on_ground'],
         ['onGround', ('right_knee'), 'on_ground']]],
    'hands_shoulder_width': [
        [['distance', ('left_hand', 'right_hand'), 'shoulder width'],
         ['relativeVAxis', ('left_hand', 'right_hand'), 'horizontal']]],
    'feet_shoulder_width': [
        [['distance', ('left_foot', 'right_foot'), 'shoulder width'],
         ['relativeVAxis', ('left_foot', 'right_foot'), 'horizontal']]],
    # ADD_SUPER_POSECODE
}


SUPER_MOTIONCODES = [

    ['body_turn_left', [('body_orientation'), 'turn_left'], False],
    ['body_turn_right', [('body_orientation'), 'turn_right'], False],
    ['body_tilt_downward', [('body_orientation'), 'tilt_downward'], True],
    ['body_tilt_upward', [('body_orientation'), 'tilt_upward'], False],
    ['body_move_left', [('body_translation'), 'move_left'], False],
    ['body_move_right', [('body_translation'), 'move_right'], False],
    ['body_move_forward', [('body_translation'), 'move_forward'], True],
    ['body_move_backward', [('body_translation'), 'move_backward'], False],
    ['body_move_upward', [('body_translation'), 'move_upward'], True],
    ['body_move_downward', [('body_translation'), 'move_downward'], False],
    ['body_move_left_forward', [('body_translation'), 'move_left_forward'], False],
    ['body_move_left_backward', [('body_translation'), 'move_left_backward'], False],
    ['body_move_right_forward', [('body_translation'), 'move_right_forward'], True],
    ['body_move_right_backward', [('body_translation'), 'move_right_backward'], False],
    ['body_move_up_left', [('body_translation'), 'move_up_left'], True],
    ['body_move_up_right', [('body_translation'), 'move_up_right'], True],
    ['body_move_down_left', [('body_translation'), 'move_down_left'], False],
    ['body_move_down_right', [('body_translation'), 'move_down_right'], False],
    # ADD_SUPER_MOTIONCODE
]

SUPER_motioncodes_REQUIREMENTS = {
    'body_turn_left': [ #yaw
        # (way 1) Significant yaw to the left
        [['PositionZ', 'orientation', 'significant_???_left']],
    ],
    'body_turn_right': [
        # (way 1) Significant yaw to the right
        [['PositionZ', 'orientation', 'significant_??right']],
    ],
    'body_tilt_downward': [
        # Significant pitch downward
        [['PositionX', 'orientation', 'significant_down']],
    ],
    'body_tilt_upward': [ #pitch
        # Significant pitch upward
        [['PositionX', 'orientation', 'significant_up']],
    ],
    'body_move_left': [
        # Significant translation along the negative x-axis
        [['translation', 'x', 'significant_left']],
    ],
    'body_move_right': [
        # Significant translation along the positive x-axis
        [['PositionX', 'translation', 'significant_right']],
    ],
    'body_move_forward': [
        # Significant translation along the negative z-axis
        [['translation', 'z', 'significant_forward']],
    ],
    'body_move_backward': [
        # Significant translation along the positive z-axis
        [['translation', 'z', 'significant_backward']],
    ],
    'body_move_upward': [
        # Significant translation along the positive y-axis
        [['translation', 'y', 'significant_upward']],
    ],
    'body_move_downward': [
        # Significant translation along the negative y-axis
        [['translation', 'y', 'significant_downward']],
    ],
    'body_move_left_forward': [
        # Significant translation along the negative x-axis and negative z-axis
        [['translation', 'x', 'significant_left'], ['translation', 'z', 'significant_forward']],
    ],
    'body_move_left_backward': [
        # Significant translation along the negative x-axis and positive z-axis
        [['translation', 'x', 'significant_left'], ['translation', 'z', 'significant_backward']],
    ],
    'body_move_right_forward': [
        # Significant translation along the positive x-axis and negative z-axis
        [['translation', 'x', 'significant_right'], ['translation', 'z', 'significant_forward']],
    ],
    'body_move_right_backward': [
        # Significant translation along the positive x-axis and positive z-axis
        [['translation', 'x', 'significant_right'], ['translation', 'z', 'significant_backward']],
    ],
    'body_move_up_left': [
        # Significant translation along the positive y-axis and negative x-axis
        [['translation', 'y', 'significant_upward'], ['translation', 'x', 'significant_left']],
    ],
    'body_move_up_right': [
        # Significant translation along the positive y-axis and positive x-axis
        [['translation', 'y', 'significant_upward'], ['translation', 'x', 'significant_right']],
    ],
    'body_move_down_left': [
        # Significant translation along the negative y-axis and negative x-axis
        [['translation', 'y', 'significant_downward'], ['translation', 'x', 'significant_left']],
    ],
    'body_move_down_right': [
        # Significant translation along the negative y-axis and positive x-axis
        [['translation', 'y', 'significant_downward'], ['translation', 'x', 'significant_right']],
    ],
    # ADD_SUPER_MOTIONCODE

}


################################################################################
#                                                                              #
#                   POSECODE SELECTION                                         #
#                                                                              #
################################################################################

# NOTE: information about posecode selection are disseminated throughout the
# code:
# - eligible posecode interpretations are defined above in section "POSECODE
#     EXTRACTION" for simplicity; 
# - rare (unskippable) & trivial posecodes were determined through satistical
#     studies (see corresponding section in captioning.py) and information was
#     later reported above; 
# - ripple effect rules based on transitive relations are directly computed
#     during the captioning process (see captioning.py).
#
# We report below information about random skip and ripple effect rules based on
# statistically frequent pairs and triplets of posecodes.

# Define the proportion of eligible (non-aggregated) posecodes that can be
# skipped for description, in average, per pose
PROP_SKIP_POSECODES = 0.30


PROP_SKIP_MOTONCODES_SPATIAL = 0.20 # 0.20
PROP_SKIP_MOTONCODES_TEMPORAL = 0.20

PROP_SKIP_TIMECODE  = 0.30

# One way to get rid of redundant posecodes is to use ripple effect rules based
# on statistically frequent pairs and triplets of posecodes. Those were obtained
# as follow:
# - automatic detection based on statistics over the dataset:
#   - general considerations:
#       - the rule involves eligible posecodes only
#       - the rule must affect at least 50 poses
#       - the rule must be symmetrically eligible for the left & right sides
#         (however, for better readability and declaration simplicity, the rules
#         are formalized below as if applying regarding to the left side only)
#   - mined relations:
#       - bi-relations (A ==> C) if the poses that have A also have C in 70% of
#         the cases
#       - tri-relations (A+B ==> C) if poses that have A+B also have C in 80% of
#         the cases, and if "A+B ==> C" is not an augmented version of a relation
#         "A ==> C" that was already mined as bi-relation.
# - manual selection of mined rules:
#   - keep only those that make sense and that should be applied whenever it is
#     possible. Other mined rules were either less meaningful; relying on weak
#     conditions (eg. when A & B were giving conditions on L body parts
#     regarding R body parts, and C was a “global” result on L body parts); or
#     pure "loopholes": when using an auxiliary posecode to get past the
#     threshold and be considered a ripple effect rule (particularly obvious
#     when A is about the upper body, and B & C are about the lower body: A
#     enabled to select a smaller set of poses for which "B ==> C" could meet
#     the rate threshold, while it could not in the entire set).
#   - split bi-directionnal bi-relations "A <==> C" split into 2 bi-relations
#     ("A ==> C" & "C ==> A"), and keep only the most meaningful (if both
#     ways were kept, both posecodes C (by applying "A ==> C") and A (by
#     applying "C ==> A") would be removed, resulting in unwanted information
#     loss).
#   - NOTE: rules that would be neutralized by previous aggregations (eg.
#     entity-based/symmetry-based) should be kept, in case such aggregations do
#     not happen (as aggregation rules could be applied at random). If not
#     considering random aggregation, such rules could be commented to reduce
#     the execution time.

# The ripple effect rules defined below can be printed in a readable way by
# copying STAT_BASED_RIPPLE_EFFECT_RULES in a python interpreter and doing the following:
# $ from tabulate import tabulate
# $ print(tabulate(STAT_BASED_RIPPLE_EFFECT_RULES, headers=["Posecode A", "Posecode B", "==> Posecode C"]))

STAT_BASED_RIPPLE_EFFECT_RULES = [
        # bi-directionnal rule
        ['[relativePosY] L hand - neck (above)', '---', '[relativePosY] L hand-shoulder (above)'],
        # uni-direction rules
        ['[angle] L knee (right angle)', '[relativeVAxis] L thigh (vertical)', '[relativePosZ] L foot - torso (behind)'],
        ['[relativePosZ] L/R foot (behind)', '[relativePosZ] L foot - torso (front)', '[relativePosZ] R foot - torso (front)'],
        ['[relativePosZ] L/R foot (front)', '[relativePosZ] L foot - torso (behind)', '[relativePosZ] R foot - torso (behind)'],
        ['[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-hip (below)', '[distance] L/R hand (wide)'],
        ['[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-hip (below)', '[relativePosY] L/R shoulder (above)'],
        ['[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-hip (below)', '[relativePosY] L/R elbow (above)'],
        ['[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-hip (below)', '[relativePosY] L/R hand (above)'],
        ['[relativePosY] L hand-hip (below)', '[relativeVAxis] R upperarm (horizontal)', '[relativePosY] L/R elbow (below)'],
        ['[relativePosY] L hand-hip (below)', '[relativeVAxis] R upperarm (horizontal)', '[relativePosY] L/R hand (below)'],
        ['[distance] L/R hand (close)', '[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-shoulder (above)'],
        ['[distance] L/R hand (close)', '[relativePosY] L hand - neck (above)', '[relativePosY] R hand-shoulder (above)'],
        ['[distance] L/R hand (close)', '[relativePosY] L hand - neck (above)', '[relativePosY] R hand - neck (above)'],
        ['[distance] L/R hand (close)', '[relativePosY] L hand-hip (below)', '[relativePosY] R hand-hip (below)'],
        ['[distance] L/R foot (close)', '[relativePosZ] L foot - torso (behind)', '[relativePosZ] R foot - torso (behind)'],
        ['[relativePosY] L/R elbow (above)', '[relativePosY] L hand-hip (below)', '[relativePosY] R hand-hip (below)'],
        ['[relativePosY] L/R hand (below)', '[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-shoulder (above)'],
        ['[relativePosY] L/R hand (below)', '[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand - neck (above)'],
        ['[relativePosY] L hand - neck (above)', '[relativePosY] R hand-hip (below)', '[distance] L/R hand (wide)'],
        ['[relativePosY] L/R hand (above)', '[relativePosY] L hand-hip (below)', '[relativePosY] R hand-hip (below)'],
        ['[relativePosY] L hand - neck (above)', '[relativePosY] R hand-hip (below)', '[relativePosY] L/R elbow (above)'],
        ['[relativePosY] L hand - neck (above)', '[relativePosY] R hand-hip (below)', '[relativePosY] L/R hand (above)'],
        ['[relativePosZ] L/R hand (front)', '[relativePosZ] L hand - torso (behind)', '[relativePosZ] R hand - torso (behind)'],
        ['[relativeVAxis] L upperarm (vertical)', '[relativeVAxis] L forearm (horizontal)', '[angle] L elbow (right angle)'],
        ['[relativeVAxis] L thigh (vertical)', '[relativeVAxis] L calf (vertical)', '[angle] L knee (straight)'],
        ['[relativePosZ] L/R knee (behind)', '[relativeVAxis] L thigh (horizontal)', '[relativePosZ] L foot - torso (behind)'],
        ['[relativePosZ] L/R knee (front)', '[relativeVAxis] L thigh (vertical)', '[relativePosZ] R foot - torso (behind)']
    ]


################################################################################
#                                                                              #
#                   POSECODE AGGREGATION                                       #
#                                                                              #
################################################################################

# Define the proportion in which an aggregation rule can be applied
PROP_AGGREGATION_HAPPENS = 0.95

# Some special textual keys to help processing
NO_VERB_KEY = '<no_verb>'
NO_VERB_SINGULAR_KEY = '<no_verb_singular>'
NO_VERB_PLURAL_KEY = '<no_verb_plural>'


MULTIPLE_SUBJECTS_KEY = '<multiple_subjects>'
JOINT_BASED_AGGREG_KEY = '<joint_based_aggreg>'

# From simple body parts to larger entities (assume side-preservation)
ENTITY_AGGREGATION = {
    ('wrist', 'elbow'):'arm',
    ('hand', 'elbow'):'arm',
    ('ankle', 'knee'):'leg',
    ('foot', 'knee'):'leg',
    ('forearm', 'upperarm'):'arm',
    ('calf', 'thigh'):'leg'}
# make it possible to query in any order
d = {(b,a):c for (a,b),c in ENTITY_AGGREGATION.items()}
ENTITY_AGGREGATION.update(d)


################################################################################
#                                                                              #
#                   POSECODE CONVERSION                                        #
#                                                                              #
################################################################################

# Helper function
def flatten_list(l):
    return [item for sublist in l for item in sublist]

# Define different ways to refer to the figure to start the description
# NOTE: use "neutral" words (unlike "he/her", which will automatically be used
# as substitutes for the word "they" at processing time, depending on the chosen
# determiner)
# MotionScript revision:
# Note 'The figure' was removed since it refers more to a static pose rather than a motion.
# Also, "They" as pronoun hence "Their" as determiner were removed due to ambiguty they bring
# into our sentences. We, as a human, cannot discreminate "they" reffering to the person or a
# set of joints.

# SENTENCE_START = ['Someone', 'The person', 'This person', 'A person', 'The body', 'The subject', 'The human', 'They']
SENTENCE_START = ['Someone', 'The person', 'This person', 'A person', 'The body', 'The subject', 'The human']

BODY_REFERENCE_MID_SENTENCE = ['the body']


# Define possible determiners (and their associated probability)
DETERMINERS = ["the", "their", "his", "her"]

# DETERMINERS_PROP = [0.5, 0.3, 0.1, 0.1]
DETERMINERS_PROP = [0.4, 0.0, 0.3, 0.3]

DETERMINER_KEY = "<determiner>"


# Define possible transitions between sentences/pieces of description (and their
# associated probability)
# Caution: some specific conditions on these transitions are used in the
# captioning pipeline. Before adding/removing any transition, one should check
# for such conditions.
TEXT_TRANSITIONS = [' while ', ', ', '. ', ' and ', ' with ']
TEXT_TRANSITIONS_With_TIME = [', ', '. ', ' and ']
TEXT_TRANSITIONS_PROP = [0.2, 0.2, 0.2, 0.2, 0.2]

SENTENCE_NAX_LENGTH = 30
PRONOUN_MAX_WORDS = 20
# Specific plural rules
PLURALIZE = {
    "foot":"feet",
    "calf":"calves"
}

# Define opposite interpretation correspondences to translate a posecode where
# "joint 1 is studied with regard to joint 2" by a posecode where "joint 2 is
# studied with regard to joint 1" (when joints are taken in reverse order, the
# posecode interpretation needs to be adapted).
# Only needed for posecodes for which the second body part (if any) matters.
OPPOSITE_CORRESP = {
    'at_right':'at_left',
    'at_left':'at_right',
    'below':'above',
    'above':'below',
    'behind':'front',
    'front':'behind',
    'close':'close',
    'shoulder width':'shoulder width',
    'spread':'spread',
    'wide':'wide'}
    # ADD_POSECODE_KIND: add interpretations if there are some new 
    # ADD_SUPER_POSECODE

OPPOSITE_CORRESP_MOTIONCODES = { # this would be used for the relative axis
                                 # motioncodes which is not implemented yet.
                                'right-to-left': 'left-to-right',
                                'above-to-below': 'below-to-above',
                                'front-to-behind': 'behind-to-front',
    }





# Define template sentences for when:
# - the description involves only one component
#     (eg. "the hands", "the right elbow", "the right hand (alone)")
# - the description involves two components
#     (eg. "the right hand"+"the left elbow"...)
#
# Format rules:
# - format "{}" into a joint name
# - format "%s" into a verb ("is"/"are")
#
# Caution when defining template sentences:
# - Template sentences must be defined for every eligible interpretation
#     (including super-posecode interpretations)
# - When necessary, use the neutral words ("their"/"them"); those will be
#     replaced by their gendered counterpart at processing time, depending on
#     the chosen determiner.
# - Keep in mind that 'is' & 'are' verbs may be removed from template sentences
#     if the "with" transition is used before.
# - Do not use random.choice in the definition of template sentences:
#     random.choice will be executed only once, when launching the program.
#     Thus, the same chosen option would be applied systematically, for all pose
#     descriptions (no actual randomization nor choice multiplicity).
#
# Interpretations that can be worded in 1 or 2-component sentences at random
# ADD_POSECODE_KIND, ADD_SUPER_POSECODE: add interpretations if there are some new 
# (currently, this is only true for distance posecodes)
OK_FOR_1CMPNT_OR_2CMPNTS = POSECODE_OPERATORS_VALUES["distance"]["category_names"]

SUBJECT_OBJECT_REQUIRED_KINDS = ['distance', 'relativePosX', 'relativePosY', 'relativePosZ']
OK_FOR_1CMPNT_OR_2CMPNTS_MOTONCODES = MOTIONCODE_OPERATORS_VALUES["proximity"]["category_names"]

subj = "{} %s"
sj = "{}"
VERB_TENSE = "<TENSE>"
# 1-COMPONENT TEMPLATE SENTENCES
ENHANCE_TEXT_1CMPNT = {
    "completely bent":
        [f"{subj} {c} bent" for c in ["completely", "fully"]] +
        [f"{subj} bent {c}" for c in ["to maximum", "to the max", "sharply"]],
    "bent more":
       [f"{subj} bent"] + [f"{subj} {c} bent" for c in ["almost completely", "rather"]],
    "right angle":
        [f"{subj} {c}" for c in ["bent at right angle", "in L-shape", "forming an L shape", "at right angle", "bent at 90 degrees", "bent at near a 90 degree angle"]],
    "bent less":
       [f"{subj} bent"] + [f"{subj} {c} bent" for c in ["partially", "partly", "rather"]],
    "slightly bent":
        [f"{subj} {c} bent" for c in ["slightly", "a bit", "barely", "nearly"]] +
        [f"{subj} bent {c}" for c in ["slightly", "a bit"]],
    "straight":
        [f"{subj} {c}" for c in ["unbent", "straight"]],
    "close":
        [f"{subj} {c}" for c in ["together", "joined", "close", "right next to each other", "next to each other"]],
    "shoulder width":
        [f"{subj} {c}" for c in ["shoulder width apart", "about shoulder width apart", "approximately shoulder width apart", "separated at shoulder width"]],
    "spread":
        [f"{subj} {c}" for c in ["apart wider than shoulder width", "further than shoulder width apart", "past shoulder width apart", "spread", "apart", "spread apart"]],
    "wide":
        [f"{subj} {c}" for c in ["spread apart", "wide apart", "spread far apart"]],
    "at_right":
        [f"{subj} {c}" for c in ["on the right", "on their right", "to the right", "to their right", "extended to the right", "turned to the right", "turned right", "reaching to the right", "out to the right", "pointing right", "out towards the right", "towards the right", "in the right direction"]],
    "at_left":
        [f"{subj} {c}" for c in ["on the left", "on their left", "to the left", "to their left", "extended to the left", "turned to the left", "turned left", "reaching to the left", "out to the left", "pointing left", "out towards the left", "towards the left", "in the left direction"]],
    "below":
        [f"{subj} {c}" for c in ["down", "lowered", "lowered down", "further down", "reaching down", "towards the ground", "towards the floor", "downwards"]],
    "above":
        [f"{subj} {c}" for c in ["up", "raised", "raised up", "reaching up", "towards the ceiling", "towards the sky", "upwards"]],
    "behind":
        [f"{subj} {c}" for c in ["in the back", f"in {DETERMINER_KEY} back", "stretched backwards", "extended back", "backwards", "reaching backward", "behind the back", "behind their back", "back"]],
    "front":
        [f"{subj} {c}" for c in ["in the front", "stretched forwards", "to the front", "reaching forward", "in front"]],
    "vertical":
        [f"{subj} {c}" for c in ["vertical", "upright", "straightened up", "straight"]],
    "horizontal":
        [f"{subj} {c}" for c in ["horizontal", "flat", "aligned horizontally", "parallel to the ground", "parallel to the floor"]],
    "bent_left":
        [f"{subj} {c} {d} left{e}" for c in ["bent to", "leaning on", "bent on", "inclined to", "angled towards"] for d in ['the', 'their'] for e in ['',' side']],
    "bent_right":
        [f"{subj} {c} {d} right{e}" for c in ["bent to", "leaning on", "bent on", "inclined to", "angled towards"] for d in ['the', 'their'] for e in ['',' side']],
    "bent_backward":
        [f"{subj} {c}" for c in ["bent backwards", "leaning back", "leaning backwards", "inclined backward", "angled backwards", "reaching backwards", "arched back"]],
    "bent_forward":
        [f"{subj} {c}" for c in ["bent forward", "leaning forwards", "bent over", "inclined forward", "angled forwards", "reaching forward", "hunched over"]],
    "kneeling":
        [f"{subj} {c}" for c in ["kneeling", "in a kneeling position", "on their knees", "on the knees"]] + [f"{d} knees are on the ground" for d in ['the', 'their']],
    "kneel_on_left":
        [f"{subj} {c}" for c in flatten_list([[f"kneeling on {d} left knee", f"kneeling on {d} left leg", f"on {d} left knee"] for d in ['the', 'their']])],
    "kneel_on_right":
        [f"{subj} {c}" for c in flatten_list([[f"kneeling on {d} right knee", f"kneeling on {d} right leg", f"on {d} right knee"] for d in ['the', 'their']])],
    # ADD_POSECODE_KIND: add template sentences for new interpretations if any 
    # ADD_SUPER_POSECODE


    # ---------     orientation_pitch
    "lying_flat_forward":
        [f"{subj} {c}" for c in ["lying flat face down", "face down on the ground", "stretched out forward on the ground"]],
    "leaning_forward":
        [f"{subj} {c}" for c in ["leaning forward", "inclined forward", "angling forward", "hunching forward"]],
    "slightly_leaning_forward":
        [f"{subj} {c}" for c in ["slightly leaning forward", "slightly inclined forward", "mildly angling forward"]],
    "neutral":
        [f"{subj} {c}" for c in ["in a neutral position", "standing straight", "upright", "not leaning"]],
    "slightly_leaning_backward":
        [f"{subj} {c}" for c in ["slightly leaning backward", "slightly inclined backward", "mildly angling backward"]],
    "leaning_backward":
        [f"{subj} {c}" for c in ["leaning backward", "inclined backward", "angling backward", "arching backward"]],
    "lying_flat_backward":
        [f"{subj} {c}" for c in ["lying flat face up", "face up on the ground", "stretched out backward on the ground"]],

    # ---------     orientation_roll
    "upside_down_right":
        [f"{subj} {c}" for c in ["upside down facing right", "turned upside down to the right"]],
    "lying_right":
        [f"{subj} {c}" for c in
         ["lying flat facing right", "stretched out to the right", "face to the right on the ground"]],
    "leaning_right":
        [f"{subj} {c}" for c in ["leaning to the right", "inclined to the right", "angling to the right"]],
    "moderately_leaning_right":
        [f"{subj} {c}" for c in ["moderately leaning to the right", "moderately inclined to the right",
                                 "moderately angling to the right"]],
    "slightly_leaning_right":
        [f"{subj} {c}" for c in
         ["slightly leaning to the right", "slightly inclined to the right", "mildly angling to the right"]],
    "slightly_leaning_left":
        [f"{subj} {c}" for c in
         ["slightly leaning to the left", "slightly inclined to the left", "mildly angling to the left"]],
    "moderately_leaning_left":
        [f"{subj} {c}" for c in
         ["moderately leaning to the left", "moderately inclined to the left", "moderately angling to the left"]],
    "leaning_left":
        [f"{subj} {c}" for c in ["leaning to the left", "inclined to the left", "angling to the left"]],
    "lying_left":
        [f"{subj} {c}" for c in
         ["lying flat facing left", "stretched out to the left", "face to the left on the ground"]],
    "upside_down_left":
        [f"{subj} {c}" for c in ["upside down facing left", "turned upside down to the left"]],

    # ---------     orientation_yaw
    "about-face_turned_clockwise":
        [f"{subj} {c}" for c in ["about-face turned clockwise", "fully rotated clockwise"]],
    "completely_turned_clockwise":
        [f"{subj} {c}" for c in ["completely turned clockwise", "360 degrees turned clockwise"]],
    "moderately_turned_clockwise":
        [f"{subj} {c}" for c in ["moderately turned clockwise", "partially rotated clockwise"]],
    "slightly_turned_clockwise":
        [f"{subj} {c}" for c in ["slightly turned clockwise", "mildly rotated clockwise"]],
    "slightly_turned_counterclockwise":
        [f"{subj} {c}" for c in ["slightly turned counterclockwise", "mildly rotated counterclockwise"]],
    "moderately_turned_counterclockwise":
        [f"{subj} {c}" for c in ["moderately turned counterclockwise", "partially rotated counterclockwise"]],
    "completely_turned_counterclockwise":
        [f"{subj} {c}" for c in ["completely turned counterclockwise", "360 degrees turned counterclockwise"]],
    "about-face_turned_counterclockwise":
        [f"{subj} {c}" for c in ["about-face turned counterclockwise", "fully rotated counterclockwise"]]
}

# 2-COMPONENT TEMPLATE SENTENCES
ENHANCE_TEXT_2CMPNTS = {
    "close":
        [f"{subj} {c} {sj}" for c in ["close to", "near to", "beside", "at the level of", "at the same level as", "right next to", "next to", "near", "joined with"]],
    "shoulder width":
        [f"{subj} {c} {sj}" for c in ["shoulder width apart from", "about shoulder width apart from", "approximately shoulder width apart from", "separated at shoulder width from"]],
    "spread":
        [f"{subj} {c} {sj}" for c in ["apart wider than shoulder width from", "further than shoulder width apart from", "past shoulder width apart from", "spread apart from", "apart from"]],
    "wide":
        [f"{subj} {c} {sj}" for c in ["wide apart from", "spread far apart from"]],
    "at_right":
        [f"{subj} {c} {sj}" for c in ["at the right of", "to the right of"]],
    "at_left":
        [f"{subj} {c} {sj}" for c in ["at the left of", "to the left of"]],
    # Simplified the 'above' and 'below' rules to increase clarity
    # Angelica: Because the other ones suggest alignment AND height,
    # for example "moves the left hand above the right hand" means one on top of the other
    "above":
        [f"{subj} {c} {sj}" for c in ["higher than", "further up than"]],
    "below":
        [f"{subj} {c} {sj}" for c in ["lower than", "further down than"]],
    "behind":
        [f"{subj} {c} {sj}" for c in ["behind", "in the back of", "located behind"]],
    "front":
        [f"{subj} {c} {sj}" for c in ["in front of", "ahead of", "located in front of"]],
    # ADD_POSECODE_KIND: add interpretations if there are some new 
    # ADD_SUPER_POSECODE
    }


##############################################################################
############################## Motion Templates ##############################
##############################################################################
VELOCITY_TERM = '<VELOCITY_TERM>'
AND_VELOCITY_TERM = '<AND_VELOCITY_TERM>'
TIME_RELATION_TERM = '<TIME_RELATION>'
INITIAL_POSE_TERM, FINAL_POSE_TERM = '<INITIAL_POSE_TERM>', '<FINAL_POSE_TERM>'
ENHANCE_TEXT_MOTION_VERBS = ['bends', 'extends', 'closes', 'spreads']
sj_obj = "<SECOND_JOINT_OBJECT>"
ENHANCE_TEXT_1CMPNT_Motion = {
    # ---------------------------------------------
    # Angular
    "significant_bend":

        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} bend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} bend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "completely"]], #"markedly", "greatly"]],

    "moderate_bend":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} bend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["", "", ]]  + #"moderately", "reasonably", "fairly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} bend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["", ""]], #"moderately", "reasonably", "fairly"]],
    "slight_bend":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} bend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit" , "just a little"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} bend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit", ]], # "just a little"]],
    "no_action":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c}{FINAL_POSE_TERM}" for c in ["not bending", "remains stationary"]],
    "slight_extension":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} extend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit", "barely", "just a little"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} extend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit"]], # "just a little"]],
    "moderate_extension":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} extend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} extend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["", ""]], # "moderately", "reasonably", "fairly"]],
    "significant_extension":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} extend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} extend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "completely"]], # "#"markedly", "greatly"]],

    # ---------------------------------------------
    #  Proximity                        ????
    "significant_closing":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in
            # [f"{intensity} close{VERB_TENSE} {t}" for t in ['together', 'to each other'] for intensity in ["significantly", "markedly", "greatly"]]+
            [f"draw{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["significantly"]] + #, "markedly", "greatly"]]+
            # [f"approach{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other' ] for intensity in ["significantly", "markedly", "greatly"]]+
            [f"come{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["significantly"]] + #, "markedly", "greatly"]]+
            # [f"join{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["significantly", "markedly", "greatly"]]
            [f"move{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["significantly"]]  #, "markedly", "greatly"]]
        ],
    "moderate_closing":

        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in
            # [f"{intensity} close{VERB_TENSE} {t}" for t in ['together', 'to each other'] for intensity in ["moderately", "reasonably", "fairly"]]+
            [f"draw{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in [""]] + #["moderately", "reasonably", "fairly"]]+
            # [f"approach{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other' ] for intensity in ["moderately", "reasonably", "fairly"]]+
            [f"come{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in [""]] + #["moderately", "reasonably", "fairly"]]+
            # [f"join{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["moderately", "reasonably", "fairly"]]+
            [f"move{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in [""]] #["moderately", "reasonably", "fairly"]]
        ],

    "slight_closing":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in
             # [f"{intensity} close{VERB_TENSE} {t}" for t in ['together', 'to each other'] for intensity in ["slightly", "a bit", "barely", "just a little"]] +
             [f"draw{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["slightly", "a bit"]] + #, "barely", "just a little"]] +
             # [f"approach{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["slightly", "a bit", "barely", "just a little"]] +
             [f"come{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["slightly", "a bit"]] + # , "barely", "just a little"]] +
             # [f"join{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["slightly", "a bit", "barely", "just a little"]] +
             [f"move{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["slightly", "a bit"]]#, "barely", "just a little"]]
         ],
    "slight_spreading":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in
             # [f"spread{VERB_TENSE} {intensity} {t}" for t in ['from one another', 'from each other', ''] for intensity in ["slightly", "a bit", "barely", "just a little"]] +
             [f"spread{VERB_TENSE} apart {intensity} {t}" for t in ['from one another', 'from each other', ''] for intensity in ["slightly", "a bit"]] + # , "barely", "just a little"]] +
             # [f"put{VERB_TENSE} {intensity} distabce between {t}" for t in ['one another', 'each other', 'themselves'] for intensity in ["slight", "a bit", "just a little"]] +
             # [f"{intensity} distance{VERB_TENSE} {t}" for t in ['from one another', 'from each other'] for intensity in ["slightly", "a bit"]] + # , "barely", "just a little"]] +
             [f"move{VERB_TENSE} {intensity} away {t}" for t in ['from each other'] for intensity in ["slightly", "a bit"]] + #"barely", "just a little"]]+
            [f"move{VERB_TENSE} {intensity} apart {t}" for t in ['from each other', ''] for intensity in ["slightly", "a bit"]] #, "barely", "just a little"]]
         ],
    "moderate_spreading":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in
             # [f"spread{VERB_TENSE} {intensity} {t}" for t in ['from one another', 'from each other', ''] for intensity in ["moderately", "reasonably", "fairly"]] +
             [f"spread{VERB_TENSE} apart {intensity} {t}" for t in ['from one another', 'from each other', ''] for intensity in[""]] + # ["moderately", "reasonably", "fairly"]] +
             # [f"put{VERB_TENSE} {intensity} distabce between {t}" for t in ['one another', 'each other', 'themselves'] for intensity in ["moderate", "reasonabld", "fair"]] +
             # [f"{intensity} distance{VERB_TENSE} {t}" for t in ['from one another', 'from each other'] for intensity in [""]] + #["moderately", "reasonably", "fairly"]] +
             [f"move{VERB_TENSE} {intensity} away {t}" for t in ['from each other'] for intensity in [""]] + #["moderately", "reasonably", "fairly"]]+
            [f"move{VERB_TENSE} {intensity} apart {t}" for t in ['from each other', ''] for intensity in [""]] #["moderately", "reasonably", "fairly"]]
         ],
    "significant_spreading":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in
             # [f"spread{VERB_TENSE} {intensity} {t}" for t in ['from one another', 'from each other', ''] for intensity in ["significantly"]] + # ["significantly", "markedly", "greatly"]] +
             [f"spread{VERB_TENSE} apart {intensity} {t}" for t in ['from one another', 'from each other', ''] for intensity in ["significantly"]] + #["significantly", "markedly", "greatly"]] +
             # [f"put{VERB_TENSE} {intensity} distabce between {t}" for t in ['one another', 'each other', 'themselves'] for intensity in ["significantly"]] + # ["significantly", "markedly", "greatly"]] +
             # [f"{intensity} distance{VERB_TENSE} {t}" for t in ['from one another', 'from each other'] for intensity in ["significantly"]] + # ["significantly", "markedly", "greatly"]] +
             [f"move{VERB_TENSE} {intensity} away {t}" for t in ['from each other'] for intensity in ["significantly"]] + # ["significantly", "markedly", "greatly"]]+
            [f"move{VERB_TENSE} {intensity} apart {t}" for t in ['from each other', ''] for intensity in ["significantly"]] # ["significantly", "markedly", "greatly"]]
         ],


    # --------------------------------------------------------------------------------
    # ------------------------Spatial Relation Templates------------------------------
    # TODO: Chek if we use the following 1-Component templates for the Spatials.
    # ----------Spatila Relation X:

    "right-to-left":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for option in
         [f"move{VERB_TENSE} from the right of {sj_obj} to the left"] +
         [f"slide{VERB_TENSE} from the right side of {sj_obj} to the left side"] +
         [f"shift{VERB_TENSE} from right to left of {sj_obj}"]],

    "left-to-right":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for option in
         [f"move{VERB_TENSE} from the left of {sj_obj} to the right"] +
         [f"slide{VERB_TENSE} from the left side of {sj_obj} to the right side"] +
         [f"shift{VERB_TENSE} from left to right of {sj_obj}"]],

    # ----------Spatila Relation Y:

    "above-to-below":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for option in
         # [f"lower{VERB_TENSE} from above {sj_obj} to {below}" for below in ['under', 'below']] +
         [f"lower{VERB_TENSE} " ] +
         # [f"drop{VERB_TENSE} from above {sj_obj} descending to {below}" for below in ['under', 'below']] +
         [f"drop{VERB_TENSE} "] +
         # [f"reach{VERB_TENSE} downward, from above {sj_obj} to {below}" for below in ['under', 'below']]],
         [f"reach{VERB_TENSE} downward"]],

    "below-to-above":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for option in
         # [f"raise{VERB_TENSE} from {below} {sj_obj} to above" for below in ['under', 'below']] +
         [f"raise{VERB_TENSE} "] +
         # [f"lift{VERB_TENSE} from {below} {sj_obj} ascending to above" for below in ['under', 'below']] +
         [f"lift{VERB_TENSE} " ] +
         # [f"reach{VERB_TENSE} upward, from {below} {sj_obj} to above" for below in ['under', 'below']]],
         [f"reach{VERB_TENSE} upward "]],

    # ----------Spatila Relation Z:

    "front-to-behind":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for option in
         # [f"move{VERB_TENSE} from the front of {sj_obj} to behind"] +
         [f"move{VERB_TENSE} to behind"] +
         # [f"transition{VERB_TENSE} from the front to a position behind {sj_obj}"] +
         [f"transition{VERB_TENSE} to a position behind "] +
         # [f"shift{VERB_TENSE} from the front to behind {sj_obj}"]],
         [f"shift{VERB_TENSE} to behind "]],
    "behind-to-front":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for option in
         # [f"advance{VERB_TENSE} from behind  of {sj_obj} to the front"] +
         [f"advance{VERB_TENSE} to in front"] +
         # [f"move{VERB_TENSE} from behind to a position in front of {sj_obj}"] +
         [f"move{VERB_TENSE} to a position in front "] +
         # [f"come{VERB_TENSE} forward, from behind to the front of {sj_obj}"]]
         [f"come{VERB_TENSE} forward "]],



    # ---------------------------------------------
    # Rotation Pitch:
    "significant_leaning_forward":
    # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE}forward{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
    # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} forward {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly"]], #, "markedly", "greatly"]],
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} forward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly"]], #, "markedly", "greatly"]],

    "moderate_leaning_forward":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} forward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} forward {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]],
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} forward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "slight_leaning_forward":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} forward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit", "just a little"]] +
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} forward {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit", "just a little"]],
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} forward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} forward {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [ "just a little bit"]],
    # "no_action":
    #     [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c}{FINAL_POSE_TERM}" for c in ["not bending", "remains stationary"]],

    "slight_leaning_backward":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} backward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} backward {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [ "just a little"]],

    "moderate_leaning_backward":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} backward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} backward {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]],
    "significant_leaning_backward":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} backward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} backward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly"]], #, "markedly", "greatly"]],

    # ---------------------------------------------
    # Rotation Roll
    "significant_leaning_right":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]],
    #                                     updated in our meeting
    "moderate_leaning_right":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} into his right {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]],
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "slight_leaning_right":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit", "barely", "just a little"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit",]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} into his right {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["just a little bit"]],

    # "no_action":
    # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c}{FINAL_POSE_TERM}" for c in ["not bending", "remains stationary"]],

    "slight_leaning_left":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} into his left {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} into his left {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["just a little"]],
    "moderate_leaning_left":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} into his left {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} into his left {c}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "significant_leaning_left":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} into his left {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} into his left {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]],
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} into his left {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly"]],

    # ---------------------------------------------
    # Rotation Yaw
    "significant_turn_clockwise":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} turn{VERB_TENSE} clockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c}  clockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly"]],
    "moderate_turn_clockwise":
            # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} turn{VERB_TENSE} clockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c} clockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "slight_turn_clockwise":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} clockwise {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [ "just a little"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c} clockwise{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit"]],

    "slight_turn_counterclockwise":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} counterclockwise {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [ "just a little"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c} counterclockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit" ]],
    "moderate_turn_counterclockwise":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} turn{VERB_TENSE} counterclockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c} counterclockwise {AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "significant_turn_counterclockwise":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} turn{VERB_TENSE} counterclockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c} counterclockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly"]],

    # ---------------------------------------------
    # Displacement X



    "very_long_right":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the right {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["far", "a great distance", "far over"]],

    "long_right":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the right {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["considerably", "noticeably", "significantly"]],

    "moderate_right":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the right {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["", ""]], #"moderately", "reasonably", "somewhat"]],

    "short_right":

            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the right {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["slightly", "a bit"]] +
            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} to the right {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in["just a little bit"]],

    "very_short_right":

            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the right {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["just a tad", "subtly"]],
    "very_short_left":

            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the left {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["just a tad", "subtly"]],

    "short_left":

            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the left {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["slightly", "a bit"]] +
            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} to the left {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in["just a little bit"]],

    "moderate_left":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the left {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["", "",]], #"moderately", "reasonably", "somewhat"]],

    "long_left":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the left {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["considerably", "noticeably", "significantly"]],

    "very_long_left":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the left {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["far", "a great distance", "far over"]],
    # --------------------------------------
    # Displacement Y
    "very_long_down":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} downwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["greatly", "far", "a great distance"]
    ],

    "long_down":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} downwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["considerably", "noticeably", "significantly"]
    ],

    "moderate_down":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} downwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["", ""] # "moderately", "reasonably", "somewhat"]
    ],

    "short_down":
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} downwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
    for verb in ["move", "shift"] for c in ["slightly", "a bit"]]+
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} downwards {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
    for verb in ["move", "shift"] for c in ["just a little bit"]],

    "very_short_down":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} downwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["just a tad", "subtly"]
    ],


    "very_short_up":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} upwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [ "just a tad", "subtly"]
    ],

    "short_up":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} upwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["slightly", "a bit"]]+
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} upwards {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["just a little bit"]],

    "moderate_up":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} upwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["", ""]#"moderately", "reasonably", "somewhat"]
    ],

    "long_up":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} upwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["considerably", "noticeably", "significantly"]
    ],

    "very_long_up":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} upwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["far", "a great distance"]
    ],

    # ----------------------------------------
    # Displacement Z
    "very_long_forward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} {dir} {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["far", "a great distance", "far over"]
        for dir in ["towards the front", "to the front"]
    ],

    "long_forward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} {dir} {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["considerably", "noticeably", "significantly"]
        for dir in ["towards the front", "to the front"]
    ],

    "moderate_forward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} {dir} {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["", ""] #"moderately", "reasonably", "somewhat"]
        for dir in ["towards the front", "to the front"]
    ],

    "short_forward":
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} {dir} {VELOCITY_TERM}{FINAL_POSE_TERM}"
    for verb in ["move", "shift"] for c in ["slightly", "a bit"]
    for dir in ["towards the front", "to the front"]]+
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {dir} {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
    for verb in ["move", "shift"] for c in ["just a little bit"]
    for dir in ["towards the front", "to the front"]
    ],

    "very_short_forward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} {dir} {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["just a tad", "subtly"]
        for dir in ["towards the front", "to the front"]
    ],



    "very_short_backward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} {dir} {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [ "just a tad", "subtly"]
        for dir in ["towards the back", "to the back"]
    ],

    "short_backward":
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} {dir} {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["slightly", "a bit"]
        for dir in ["towards the back", "to the back"]]+
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {dir} {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["just a little bit"]
        for dir in ["towards the back", "to the back"]
    ],

    "moderate_backward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} {dir} {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["", ""] #"moderately", "reasonably", "somewhat"]
        for dir in ["towards the back", "to the back"]
    ],

    "long_backward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} {dir} {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["considerably", "noticeably", "significantly"]
        for dir in ["towards the back", "to the back"]
    ],

    "very_long_backward":
    [
         f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} {dir} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for verb in ["move", "shift"] for c in ["far", "a great distance", "far over"]
         for dir in ["towards the back", "to the back"]
    ],

}

x=[
    f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} {dir} {VELOCITY_TERM}{FINAL_POSE_TERM}"
    for verb in ["move", "shift"] for c in ["far", "a great distance", "far over"] for dir in ["towards the back", "to the back"]
]





ENHANCE_TEXT_2CMPNT_Motion = {

    # ---------------------------------------------
    # Proximity:

    "significant_closing":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"draw{VERB_TENSE} significantly near to",  f"come{VERB_TENSE} significantly closer to"]],
    "moderate_closing":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"move{VERB_TENSE} nearer to", f"get{VERB_TENSE} closer to"]],
    "slight_closing":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [ f"get{VERB_TENSE} a bit closer to", f"move{VERB_TENSE} a bit closer to"]],
    "stationary":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"remaine{VERB_TENSE} stationary with", f"stay{VERB_TENSE} still with", f"keep{VERB_TENSE} the same distance from"]],
    "slight_spreading":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"spread{VERB_TENSE} a bit apart from", f"move{VERB_TENSE} a bit away from"]],
    "moderate_spreading":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"spread{VERB_TENSE} away from", f"move{VERB_TENSE} farther from"]],
    "significant_spreading":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"spread{VERB_TENSE} significantly apart from", f"move{VERB_TENSE} significantly away from"]],

    # --------------------------------------------------------------------------------
    # ------------------------Spatial Relation Templates------------------------------

    # ----------Spatila Relation X:

    "right-to-left":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for option in
         [f"move{VERB_TENSE} from the right side of {sj_obj} to the left side of {sj_obj}" ] +
         [f"shift{VERB_TENSE} from the right side of {sj_obj} to the left side of {sj_obj}" ]],

    "left-to-right":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for option in
        [f"move{VERB_TENSE} from the left side of {sj_obj} to the right side of {sj_obj}" ] +
        [f"shift{VERB_TENSE} from the left side of {sj_obj} to the right side of {sj_obj}" ]],



    # ----------Spatila Relation Y:

    "above-to-below":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for option in
         [f"lower{VERB_TENSE} from above {sj_obj} to {below} {sj_obj}" for below in ['under', 'below']] +
         [f"drop{VERB_TENSE} from above {sj_obj} descending to {below} {sj_obj}" for below in ['under', 'below']] +
         [f"reach{VERB_TENSE} downward, from above {sj_obj} to {below} {sj_obj}" for below in ['under', 'below']]],

    "below-to-above":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for option in
         [f"raise{VERB_TENSE} from {below} {sj_obj} to above {sj_obj}" for below in ['under', 'below']] +
         [f"lift{VERB_TENSE} from {below} {sj_obj} ascending to above {sj_obj}" for below in ['under', 'below']] +
         [f"reach{VERB_TENSE} upward, from {below} {sj_obj} to above {sj_obj}" for below in ['under', 'below']]],


    # ----------Spatila Relation Z:

    "front-to-behind":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for option in
         [f"move{VERB_TENSE} from in front of {sj_obj} to behind {sj_obj}" ] +
         [f"transition{VERB_TENSE} from in front of {sj_obj} to a position behind {sj_obj}" ] +
         [f"shift{VERB_TENSE} from in front of {sj_obj} to behind {sj_obj}" ]],

    "behind-to-front":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {option} {VELOCITY_TERM}{FINAL_POSE_TERM}"
         for option in
         [f"advance{VERB_TENSE} from behind {sj_obj} to in front of {sj_obj}" ] +
         [f"move{VERB_TENSE} from behind {sj_obj} to a position in front of {sj_obj}" ] +
         [f"come{VERB_TENSE} forward, from behind {sj_obj} to in front of {sj_obj}" ]]
}







# Todo: define enhance with start or end pose description

# VELOCITY_ADJECTIVES = {
#     'very_slow': ['super slow', 'exceptionally sluggish', 'gradual', 'leisurely', 'sluggish'],
#     'slow': ['slowly', 'steadily', 'slowly and steadily', 'very slowly', 'gently', 'at a slow pace', 'at an easy pace'],
#     'moderate': ['at a moderate pace',  'smoothly', 'in a balanced manner', 'at a normal speed',
#                  'at a steady rate', 'at an average pace', 'with no rush', 'at a normal speed', 'at a medium pace'],
#     'fast': ['quickly', 'rapidly', 'speedily', 'briskly'],
#     'very_fast': ['in a flash', 'like lightning', 'very fast',  'crazy fast', 'rapidly', 'lightning-quick', 'super fast'],
# }
VELOCITY_ADJECTIVES = {
     'very_slow': ['very slowly', 'extremely slowly'],
     'slow': ['slowly', 'steadily', 'gently', 'at a slow pace', 'leisurely', 'gradually'],
     'moderate': [''],
     'fast': ['quickly', 'rapidly', 'speedily', 'briskly'],
     'very_fast': ['very fast',  'extremely fast', 'very rapidly', 'very quickly'],
 }











# CHRONOLOGICAL_ORDER_ADJECTIVE = {
#     "preceding_a_moment": ["well before", "long before", "in advance", "significantly earlier", "prior to this moment"],
#      "soon_before": ["just before", "shortly before", "not long before", "a moment earlier", "right before"],
#      "shortly_before": ["just moments before", "a few seconds earlier", "moments prior", "not too long before", "in the seconds leading up to"],
#      "immediately_before": ["right before", "just before", "in the very last moment", "in the seconds right before", "in the split second before"],
#      "simultaneously": ["at the same time", "simultaneously", "concurrently", "meantime", "meanwhile"],
#      "immediately_after": ["right after", "just after", "in the following moment", "a second later", "immediately afterward"],
#      "shortly_after": ["just seconds afterward", "a moment later", "shortly after", "not long after", "in the seconds following"],
#      "soon_after": ["in a matter of seconds", "within a short time", "shortly thereafter", "soon after", "a brief moment later"],
#      "after_a_moment": ["after a while", "after some time", "following a delay", "eventually", "after a moment"]
# }
CHRONOLOGICAL_ORDER_ADJECTIVE = {
     "preceding_a_moment": ["much earlier", "significantly earlier"],
      "soon_before": ["just before", "shortly beforehand", "a moment earlier"],
      "shortly_before": ["just moments before", "a few seconds earlier", "moments prior", "not too long before"],
      "immediately_before": ["right before", "just before", "in the second right before"],
      "simultaneously": ["at the same time", "simultaneously", "in the meantime", "meanwhile"],
      "immediately_after": ["right after", "immediately after", "a second later"],
      "shortly_after": ["a moment later", "a few seconds later", "shortly after", "not long after"],
      "soon_after": ["shortly after", "soon after", "a moment later"],
      "after_a_moment": ["after a while", "after some time", "eventually"]
 }




INITIAL_STATE_TRANSITIONS = [
    # "initially, ",
    # "in the beginning, ",
    # "in the initial pose, ",
    # "in the starting position, ",
    # "at the outset, ",
    # "originally, ",
    # "in the initial stance, ",
    # "in the initial configuration, ",
    # "in the early position, ",
    # "in the initial frame, "
    # "to start with, ",
    # "in the beginning, ",
    # "to begin with, ",
    # "at the start, ",
    # "first, ",
    # "at the beginning "
    # "<INIT_STATE1>",
    # "<INIT_STATE2>"
    ' ',
    ' ',
]


FINAL_STATE_TRANSITIONS = [
    # " afterward, ",
    # " subsequently, ",
    # " following that, ",
    # " in the end, ",
    # " in the final pose, ",
    # " in the ending position, ",
    # " at the conclusion, ",
    # " in the last pose, ",
    # " in the ultimate position, ",
    ' ',
    ' '

]


# pose2action_transitions = [ "from this position, ",
#                             "having assumed that pose, ",
#                             "with that alignment, ",
#                             "from that stance, ",
#                             "subsequently, ",
#                             "then, " # more often than the others.
#                           ]
pose2action_transitions = [ "from this position, ",
                             "with that pose, ",
                             "from that pose, ",
                             "from this stance, ",
                             "then, " # more often than the others.
                           ]



action2pose_transitions = [ "ending up in ",
                            "resulting in ",
                            "concluding with ",
                            "this leads to ",
                            "transitioning to ",
                            "finally, ",
                          ]
# Do not to forget add your irregular verbs here
SPELLING_CORRECTIONS = [ ('moveing', 'moving'),
                         ('distanceing', 'distancing'),
                         ('geting', 'getting'),
                         ('puting', 'putting'),
                         ('closeing', 'closing'),
                         ('reachs', 'reaches'),

                         # Due to a bug in poescropts
                         ('hisleft', 'his left'),
                         ('hisright', 'his right'),
                         ('herleft', 'her left'),
                         ('herright', 'her right')

                       ]



# This variable controls the length time window as well
# as the portion of captions for each sample
AUGMENTATION_LENGTH = 2.0  # in seconds
AUGMENTATION_PORTION = 1

ablation = ['nothing for now']

import captioning_data_ablation as cpd

if 'intensity' in ablation:
    ENHANCE_TEXT_1CMPNT_Motion = cpd.ENHANCE_TEXT_1CMPNT_Motion__ABLATION_INTENSITY
    ENHANCE_TEXT_2CMPNT_Motion = cpd.ENHANCE_TEXT_2CMPNT_Motion__ABLATION_INTENSITY
if 'velocity' in ablation:
    VELOCITY_ADJECTIVES = cpd.VELOCITY_ADJECTIVES_ABLATION
if 'chronological' in ablation:
    CHRONOLOGICAL_ORDER_ADJECTIVE = cpd.CHRONOLOGICAL_ORDER_ADJECTIVE__ABLATION
print()