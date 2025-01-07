import os

TRAIN_DIR = "../antrenare"

FACES_DIR = os.path.join(TRAIN_DIR, "faces")
NEGATIVE_DIR = os.path.join(TRAIN_DIR, "negatives")
IMPORTANT_FILES_DIR = "../important_files"
SMALL_SET_DIR = os.path.join(FACES_DIR, "small_set")
BIG_SET_DIR = os.path.join(FACES_DIR, "all")
FAKE_TEST_DIR = "../evaluare/fake_test"
VALIDATION_DIR = "../validare/"
ALL_FACES_DIR = "all/"

CHARACTERS = ["dexter", "deedee", "dad", "mom", "unknown"]
SHAPES_IN_ALL = 6

ALL_DESCRIPTORS_WIDTH = [3, 4, 9, 8, 8, 9]
ALL_DESCRIPTORS_HEIGHT = [4, 5, 10, 7, 6, 6]
ALL_DESCRIPTORS = [9, 7, 5, 6, 6, 5]

NR_CHARACTER_PHOTOS = 1000

AVG_FACE_WIDTH = 133
AVG_FACE_HEIGHT = 118
STD_IMG_SHAPE = (360, 480, 3)
LOWEST_FACE_VARIANCE = 790
AVG_FACE_VARIANCE = 4700

RANDOM_IMG_GEN_TRIES = 10
