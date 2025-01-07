import os
from constants import *

class Parameters:
    def __init__(self):
        self.base_dir = "../"
        self.dir_pos_examples = BIG_SET_DIR 
        self.dir_neg_examples = NEGATIVE_DIR
        self.dir_test_examples = os.path.join(VALIDATION_DIR, "validare_20")# 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.path_annotations = os.path.join(VALIDATION_DIR, "task1_gt_validare_20.txt")
        self.dir_save_files = os.path.join(self.base_dir, 'saved_files')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.hog_cell_width = 6  # dimensiunea celulei
        self.hog_cell_height = 6  # dimensiunea celulei

        self.descriptors = 6

        self.window_width = self.hog_cell_width * self.descriptors
        self.window_height = self.hog_cell_height * self.descriptors

        self.dim_descriptor_cell = self.hog_cell_height * self.hog_cell_width  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 6713  # numarul exemplelor pozitive
        self.number_negative_examples = 10000  # numarul exemplelor negative
        self.has_annotations = False
        self.threshold = 0