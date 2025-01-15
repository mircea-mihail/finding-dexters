import numpy as np
import matplotlib.pyplot as plt
import os

from constants import *
from visualisation import *
from character_classes import *
from build_dataset import *
from FacialDetector import *

def run_classifier_for_all(test_examples, test_annotations):
    params: Parameters = Parameters(ALL_DESCRIPTORS_WIDTH, ALL_DESCRIPTORS_HEIGHT, ALL_DESCRIPTORS, BIG_SET_DIR, "all")
    params.number_negative_examples = ALL_NEGATIVE_EXAMPLES  # numarul exemplelor negative
    params.number_positive_examples = ALL_POSITIVE_EXAMPLES
    params.current_character = "all"
    
    params.threshold = -1 # toate ferestrele cu scorul > threshold si maxime locale devin detectii
    params.has_annotations = True

    params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
    params.use_flip_images = True  # adauga imaginile cu fete oglindite

    if params.use_flip_images:
        params.number_positive_examples = [v * 2 for v in params.number_positive_examples]

    params.dir_test_examples = test_examples
    params.path_annotations = test_annotations

    all_facial_detector: FacialDetector = FacialDetector(params)
    all_facial_detector.best_models = []

    # load positives
    positive_features = []
    for i in range(len(params.descriptors)):
        positive_features_path = os.path.join(ALL_BEST_MODEL_DIR, 'all_positive_descriptors_' + 
                                str(params.hog_cell_widths[i]) + 'X' + str(params.hog_cell_heights[i]) + '_' +
                                str(params.number_positive_examples[i]) + '.npy')
        if os.path.exists(positive_features_path):
            positive_features.append(np.load(positive_features_path))
            print('loaded positive example features')
        else:
            print('check positive example feature path')
            print(positive_features_path )

    # load negatives
    negative_features = []
    for i in range(len(params.descriptors)):
        negative_features_path = os.path.join(ALL_BEST_MODEL_DIR, 'all_negative_descriptors_' + 
                                str(params.hog_cell_widths[i]) + 'X' + str(params.hog_cell_heights[i]) + '_' + 
                                str(params.number_negative_examples) + '.npy')
        if os.path.exists(negative_features_path):
            negative_features.append(np.load(negative_features_path))
            print('loaded negative example features')
        else:
            print('check negative example feature path')

    # load liniar classifier
    for i in range(len(positive_features)):
        svm_file_name = os.path.join(ALL_BEST_MODEL_DIR, 'best_model_%dX%d_%d_%d_%d' %
                                        (all_facial_detector.params.hog_cell_widths[i], all_facial_detector.params.hog_cell_heights[i], all_facial_detector.params.descriptors[i],
                                        all_facial_detector.params.number_negative_examples, all_facial_detector.params.number_positive_examples[i]))
        if os.path.exists(svm_file_name):
            all_facial_detector.best_models.append(pickle.load(open(svm_file_name, 'rb')))  
            print('loaded liniar classifier')
        else:
            print("check classifier path")

    detections, scores, file_names = all_facial_detector.run()

    if params.has_annotations:
        all_facial_detector.eval_detections(detections, scores, file_names)
        show_detections_with_ground_truth(detections, scores, file_names, params)
    else:
        show_detections_without_ground_truth(detections, scores, file_names, params)