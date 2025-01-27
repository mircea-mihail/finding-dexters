import os

TRAIN_DIR = "../antrenare"

FACES_DIR = os.path.join(TRAIN_DIR, "faces")
NEGATIVE_DIR = os.path.join(TRAIN_DIR, "negatives")
HARD_MINED_DIR = os.path.join(TRAIN_DIR, "hard_mined_negatives")
DEXTER_HARD_MINED_DIR = os.path.join(HARD_MINED_DIR, "custom-characters/for-dexter")
IMPORTANT_FILES_DIR = "../important_files"
SMALL_SET_DIR = os.path.join(FACES_DIR, "small_set")
BIG_SET_DIR = os.path.join(FACES_DIR, "all")
FAKE_TEST_DIR = "../evaluare/fake_test"
VALIDATION_DIR = "../validare/"
ALL_FACES_DIR = "all/"

CHARACTERS = ["dexter", "deedee", "dad", "mom", "unknown"]
SHAPES_IN_ALL = 6

# ------------------------------------- ALL CRUCIAL
ALL_BEST_MODEL_DIR = "../best_models/all"
ALL_DESCRIPTORS_WIDTH =  [4, 9, 8, 8, 9]
ALL_DESCRIPTORS_HEIGHT = [5, 10, 7, 6, 6]
ALL_DESCRIPTORS =        [7, 5, 6, 6, 5]
ALL_POSITIVE_EXAMPLES = [1024, 1180, 1362, 981, 1266] 
ALL_NEGATIVE_EXAMPLES = 12311

# ------------------------------------- DEXTER
DEXTER_BEST_MODEL_DIR = "../best_models/all"
DEXTER_DESCRIPTORS_WIDTH =  [7, 8]
DEXTER_DESCRIPTORS_HEIGHT = [6, 6]
DEXTER_DESCRIPTORS =        [6, 6]
# ALL_POSITIVE_EXAMPLES = [1024, 1180, 1362, 981, 1266] 
# ALL_NEGATIVE_EXAMPLES = 12311

NR_CHARACTER_PHOTOS = 1000

AVG_FACE_WIDTH = 133
AVG_FACE_HEIGHT = 118
STD_IMG_SHAPE = (360, 480, 3)
LOWEST_FACE_VARIANCE = 790
AVG_FACE_VARIANCE = 4700

RANDOM_IMG_GEN_TRIES = 10

CELLS_PER_BLOCK = (2, 2)
MEDIAN_SIZE = 3

MIN_VALUE = 60
MAX_VALUE = 235

MIN_SATURATION = 30
MAX_SATURATION = 185