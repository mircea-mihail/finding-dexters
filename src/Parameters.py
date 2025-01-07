import os
from constants import *

class Parameters:
    def __init__(self, hog_widths, hog_heights, descriptors, positive_dir):
        self.base_dir = "../"
        self.dir_pos_examples = positive_dir
        self.dir_neg_examples = NEGATIVE_DIR
        self.overlap = 0.8

        # self.dir_test_examples = os.path.join(VALIDATION_DIR, "validare")# 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        # self.path_annotations = os.path.join(VALIDATION_DIR, "task1_gt_validare.txt")
 
        self.dir_test_examples = os.path.join(VALIDATION_DIR, "validare_20")# 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.path_annotations = os.path.join(VALIDATION_DIR, "task1_gt_validare_20.txt")
        self.dir_save_files = os.path.join(self.base_dir, 'saved_files')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.hog_cell_widths = hog_widths  # dimensiunea celulei
        self.hog_cell_heights = hog_heights  # dimensiunea celulei
        self.descriptors = descriptors

        self.window_widths = [self.hog_cell_widths[i] * self.descriptors[i] for i in range(len(self.descriptors))] 
        self.window_heights = [self.hog_cell_heights[i] * self.descriptors[i] for i in range(len(self.descriptors))]

        self.positive_dir_names = [f"{self.hog_cell_widths[i]}X{self.hog_cell_heights[i]}" for i in range(len(self.descriptors))]
        self.number_positive_examples = [len(os.listdir(os.path.join(self.dir_pos_examples, dir))) for dir in self.positive_dir_names]
        self.number_negative_examples = [] # numarul exemplelor negative pt fiecare forma
        self.has_annotations = False
        self.threshold = 0
        
        # try? 
        #   cells_per_block = (3, 3)
