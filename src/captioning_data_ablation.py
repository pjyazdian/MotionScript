from text2pose.posescript.captioning_data import *
ENHANCE_TEXT_1CMPNT_Motion__ABLATION_INTENSITY = {
    # ---------------------------------------------
    # Angular
    "significant_bend":

        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} bend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} bend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]], #"markedly", "greatly"]],

    "moderate_bend":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} bend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["", "", ]]  + #"moderately", "reasonably", "fairly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} bend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["", ""]], #"moderately", "reasonably", "fairly"]],
    "slight_bend":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} bend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit" , "just a little"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} bend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]], # "just a little"]],
    "no_action":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c}{FINAL_POSE_TERM}" for c in ["not bending", "remains stationary"]],
    "slight_extension":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} extend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit", "barely", "just a little"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} extend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]], # "just a little"]],
    "moderate_extension":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} extend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} extend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["", ""]], # "moderately", "reasonably", "fairly"]],
    "significant_extension":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} extend{VERB_TENSE}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} extend{VERB_TENSE} {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]], # "#"markedly", "greatly"]],

    # ---------------------------------------------
    #  Proximity                        ????
    "significant_closing":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in
            # [f"{intensity} close{VERB_TENSE} {t}" for t in ['together', 'to each other'] for intensity in ["significantly", "markedly", "greatly"]]+
            [f"draw{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in [""]] + #, "markedly", "greatly"]]+
            # [f"approach{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other' ] for intensity in ["significantly", "markedly", "greatly"]]+
            [f"come{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in [""]] + #, "markedly", "greatly"]]+
            # [f"join{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["significantly", "markedly", "greatly"]]
            [f"move{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in [""]]  #, "markedly", "greatly"]]
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
             [f"draw{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in [""]] + #, "barely", "just a little"]] +
             # [f"approach{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["slightly", "a bit", "barely", "just a little"]] +
             [f"come{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in [""]] + # , "barely", "just a little"]] +
             # [f"join{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in ["slightly", "a bit", "barely", "just a little"]] +
             [f"move{VERB_TENSE} {intensity} {t}" for t in ['together', 'to each other'] for intensity in [""]]#, "barely", "just a little"]]
         ],
    "slight_spreading":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in
             # [f"spread{VERB_TENSE} {intensity} {t}" for t in ['from one another', 'from each other', ''] for intensity in ["slightly", "a bit", "barely", "just a little"]] +
             [f"spread{VERB_TENSE} apart {intensity} {t}" for t in ['from one another', 'from each other', ''] for intensity in [""]] + # , "barely", "just a little"]] +
             # [f"put{VERB_TENSE} {intensity} distabce between {t}" for t in ['one another', 'each other', 'themselves'] for intensity in ["slight", "a bit", "just a little"]] +
             # [f"{intensity} distance{VERB_TENSE} {t}" for t in ['from one another', 'from each other'] for intensity in ["slightly", "a bit"]] + # , "barely", "just a little"]] +
             [f"move{VERB_TENSE} {intensity} away {t}" for t in ['from each other'] for intensity in [""]] + #"barely", "just a little"]]+
            [f"move{VERB_TENSE} {intensity} apart {t}" for t in ['from each other', ''] for intensity in [""]] #, "barely", "just a little"]]
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
             [f"spread{VERB_TENSE} apart {intensity} {t}" for t in ['from one another', 'from each other', ''] for intensity in [""]] + #["significantly", "markedly", "greatly"]] +
             # [f"put{VERB_TENSE} {intensity} distabce between {t}" for t in ['one another', 'each other', 'themselves'] for intensity in ["significantly"]] + # ["significantly", "markedly", "greatly"]] +
             # [f"{intensity} distance{VERB_TENSE} {t}" for t in ['from one another', 'from each other'] for intensity in ["significantly"]] + # ["significantly", "markedly", "greatly"]] +
             [f"move{VERB_TENSE} {intensity} away {t}" for t in ['from each other'] for intensity in [""]] + # ["significantly", "markedly", "greatly"]]+
            [f"move{VERB_TENSE} {intensity} apart {t}" for t in ['from each other', ''] for intensity in [""]] # ["significantly", "markedly", "greatly"]]
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
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} forward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]], #, "markedly", "greatly"]],

    "moderate_leaning_forward":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} forward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} forward {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]],
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} forward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "slight_leaning_forward":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} forward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit", "just a little"]] +
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} forward {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit", "just a little"]],
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} forward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} forward {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [ ""]],
    # "no_action":
    #     [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c}{FINAL_POSE_TERM}" for c in ["not bending", "remains stationary"]],

    "slight_leaning_backward":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} backward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} backward {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [ ""]],

    "moderate_leaning_backward":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} backward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} backward {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]],
    "significant_leaning_backward":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} backward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} backward {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]], #, "markedly", "greatly"]],

    # ---------------------------------------------
    # Rotation Roll
    "significant_leaning_right":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    #                                     updated in our meeting
    "moderate_leaning_right":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} into his right {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]],
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "slight_leaning_right":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["slightly", "a bit", "barely", "just a little"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} into his right {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["",]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} into his right {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],

    # "no_action":
    # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c}{FINAL_POSE_TERM}" for c in ["not bending", "remains stationary"]],

    "slight_leaning_left":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} into his left {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} into his left {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "moderate_leaning_left":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} into his left {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} into his left {c}{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "significant_leaning_left":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} lean{VERB_TENSE} into his left {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} into his left {c}{AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]],
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} lean{VERB_TENSE} {c} into his left {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],

    # ---------------------------------------------
    # Rotation Yaw
    "significant_turn_clockwise":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} turn{VERB_TENSE} clockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c}  clockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "moderate_turn_clockwise":
            # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} turn{VERB_TENSE} clockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c} clockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "slight_turn_clockwise":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} clockwise {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [ ""]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c} clockwise{VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],

    "slight_turn_counterclockwise":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} counterclockwise {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [ ""]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c} counterclockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["" ]],
    "moderate_turn_counterclockwise":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} turn{VERB_TENSE} counterclockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["moderately", "reasonably", "fairly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c} counterclockwise {AND_VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],
    "significant_turn_counterclockwise":
        # [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} turn{VERB_TENSE} counterclockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in ["significantly", "markedly", "greatly"]] +
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} turn{VERB_TENSE} {c} counterclockwise {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [""]],

    # ---------------------------------------------
    # Displacement X



    "very_long_right":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the right {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in [""]],

    "long_right":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the right {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in [""]],

    "moderate_right":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the right {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["", ""]], #"moderately", "reasonably", "somewhat"]],

    "short_right":

            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the right {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in [""]] +
            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} to the right {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in[""]],

    "very_short_right":

            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the right {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in [""]],
    "very_short_left":

            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the left {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in [""]],

    "short_left":

            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the left {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in [""]] +
            [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} to the left {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in[""]],

    "moderate_left":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the left {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in ["", "",]], #"moderately", "reasonably", "somewhat"]],

    "long_left":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the left {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in [""]],

    "very_long_left":
        [
            f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} to the left {VELOCITY_TERM}{FINAL_POSE_TERM}"
            for verb in ["move", "shift"] for c in [""]],
    # --------------------------------------
    # Displacement Y
    "very_long_down":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} downwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]
    ],

    "long_down":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} downwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]
    ],

    "moderate_down":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} downwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["", ""] # "moderately", "reasonably", "somewhat"]
    ],

    "short_down":
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} downwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
    for verb in ["move", "shift"] for c in [""]]+
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} downwards {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
    for verb in ["move", "shift"] for c in [""]],

    "very_short_down":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} downwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]
    ],


    "very_short_up":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} upwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [ ""]
    ],

    "short_up":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} upwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]]+
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} upwards {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]],

    "moderate_up":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} upwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["", ""]#"moderately", "reasonably", "somewhat"]
    ],

    "long_up":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} upwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]
    ],

    "very_long_up":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} upwards {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]
    ],

    # ----------------------------------------
    # Displacement Z
    "very_long_forward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} forward {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]
    ],

    "long_forward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} forward {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]
    ],

    "moderate_forward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} forward {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["", ""] #"moderately", "reasonably", "somewhat"]
    ],

    "short_forward":
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} forward {VELOCITY_TERM}{FINAL_POSE_TERM}"
    for verb in ["move", "shift"] for c in [""]]+
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} forward {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
    for verb in ["move", "shift"] for c in [""]],

    "very_short_forward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} forward {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]
    ],



    "very_short_backward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} backward {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [ ""]],

    "short_backward":
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} backward {VELOCITY_TERM}{FINAL_POSE_TERM}"
    for verb in ["move", "shift"] for c in [""]]+
    [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} backward {c} {AND_VELOCITY_TERM}{FINAL_POSE_TERM}"
    for verb in ["move", "shift"] for c in [""]],

    "moderate_backward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} backward {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in ["", ""] #"moderately", "reasonably", "somewhat"]
    ],

    "long_backward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} backward {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]
    ],

    "very_long_backward":
    [
        f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {verb}{VERB_TENSE} {c} backward {VELOCITY_TERM}{FINAL_POSE_TERM}"
        for verb in ["move", "shift"] for c in [""]
    ],

}


ENHANCE_TEXT_2CMPNT_Motion__ABLATION_INTENSITY = {

    # ---------------------------------------------
    # Proximity:

    "significant_closing":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"draw{VERB_TENSE} near to",  f"come{VERB_TENSE} closer to"]],
    "moderate_closing":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"move{VERB_TENSE} nearer to", f"get{VERB_TENSE} closer to"]],
    "slight_closing":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [ f"get{VERB_TENSE} closer to", f"move{VERB_TENSE} closer to"]],
    "stationary":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"remaine{VERB_TENSE} stationary with", f"stay{VERB_TENSE} still with", f"keep{VERB_TENSE} the same distance from"]],
    "slight_spreading":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"spread{VERB_TENSE} apart from", f"move{VERB_TENSE} away from"]],
    "moderate_spreading":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"spread{VERB_TENSE} away from", f"move{VERB_TENSE} farther from"]],
    "significant_spreading":
        [f"{TIME_RELATION_TERM}{INITIAL_POSE_TERM}{subj} {c} {sj_obj} {VELOCITY_TERM}{FINAL_POSE_TERM}" for c in [f"spread{VERB_TENSE} apart from", f"move{VERB_TENSE} away from"]],

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


VELOCITY_ADJECTIVES_ABLATION = {
     'very_slow': [''],
     'slow': [''],
     'moderate': [''],
     'fast': [''],
     'very_fast': [''],
 }



CHRONOLOGICAL_ORDER_ADJECTIVE__ABLATION = {
     "preceding_a_moment": [""],
      "soon_before": [""],
      "shortly_before": [""],
      "immediately_before": [""],
      "simultaneously": [""],
      "immediately_after": [""],
      "shortly_after": [""],
      "soon_after": [""],
      "after_a_moment": [""]
 }
 
 