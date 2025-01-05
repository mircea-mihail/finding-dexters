import os

TRAIN_DIR = "../antrenare"

FACES_DIR = os.path.join(TRAIN_DIR, "faces")
NEGATIVE_DIR = os.path.join(TRAIN_DIR, "negatives")
IMPORTANT_FILES_DIR = "../important_files"
SMALL_SET_DIR = os.path.join(FACES_DIR, "small_set")
BIG_SET_DIR = os.path.join(FACES_DIR, "all")
FAKE_TEST_DIR = "../evaluare/fake_test"
VALIDATION_DIR = "../validare/"

CHARACTERS = ["dexter", "deedee", "dad", "mom", "unknown"]

NR_CHARACTER_PHOTOS = 1000

AVG_FACE_WIDTH = 133
AVG_FACE_HEIGHT = 118
STD_IMG_SHAPE = (360, 480, 3)
LOWEST_FACE_VARIANCE = 790
AVG_FACE_VARIANCE = 4700

RANDOM_IMG_GEN_TRIES = 10
