import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import ntpath
import glob

from Parameters import *
from constants import *
from general_utility import *

def print_square_info():
    train_notation_files = [file for file in os.listdir(TRAIN_DIR) if os.path.splitext(file)[1] == ".txt"]

    avg_width = 0
    avg_height = 0
    items = 0

    rectangle_ratios = {}
    rectangle_ratios["dad"] = []
    rectangle_ratios["deedee"] = []
    rectangle_ratios["dexter"] = []
    rectangle_ratios["mom"] = []
    rectangle_ratios["unknown"] = []

    for file in train_notation_files:
        df = pd.read_csv(os.path.join(TRAIN_DIR, file), header=None)

        for i in range(df.shape[0]): 
            row = (df.iloc[i]).tolist()[0].split(" ")
            # get ox / oy
            width = int(row[3]) - int(row[1])
            height = int(row[4]) - int(row[2])
            ratio = width / height
            rectangle_ratios[row[5]].append(ratio)

            avg_width += width 
            avg_height += height
            items += 1


    print("avg width", round(avg_width/items), "and height", round(avg_height/items), "and avg_area", round(avg_width*avg_height/items/items))

    # Plot the data in each subplot individually
    # axes[0, 0].scatter(ox, rectangle_ratios["dad"])
    num_bins = 25
    lim = 2.5

    for character in CHARACTERS:
        bin_edges = np.linspace(0, lim, num_bins)
        counts, edges = np.histogram(rectangle_ratios[character], bins=bin_edges)

        plt.hist(rectangle_ratios[character], bins=bin_edges, edgecolor='black', alpha=0.7)
        plt.title(character + " argmax " +  str(edges[np.argmax(counts)]))
        plt.xlabel("Ratio Value")
        plt.ylabel("Frequency")
        plt.show()

def print_img(img):
    cv.imshow('Image', img)
    cv.waitKey(0)  # Wait for a key press
    cv.destroyAllWindows()

def inspect_photos():
    for character in CHARACTERS[:4]:
        for i in range(NR_CHARACTER_PHOTOS):
            read_photo(i, character)    

def get_avg_variance():
    all = 0
    avg_var = 0
    lowest_var = np.Infinity
    for character in CHARACTERS:
        face_names = os.listdir(os.path.join(FACES_DIR, character))
        for face_name in face_names:
            face = cv.imread(os.path.join(os.path.join(FACES_DIR, character), face_name))
            var = np.var(face)
            avg_var += var
            all += 1
            if lowest_var > var:
                lowest_var = var
    
    print(f"lowest var: {lowest_var}, avg_var: {avg_var/all}, photos checked: {all}")

#from the lab

def show_detections_without_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        print('Apasa orice tasta pentru a continua...')
        cv.imshow('image', np.uint8(image))
        cv.waitKey(0)


def show_detections_with_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate. Deseneaza bounding box-urile prezice si cele corecte.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """

    ground_truth_bboxes = np.loadtxt(params.path_annotations, dtype='str')
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        annotations = ground_truth_bboxes[ground_truth_bboxes[:, 0] == short_file_name]

        # show ground truth bboxes
        for detection in annotations:
            cv.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])), (0, 255, 0), thickness=1)

        cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        # print('Apasa orice tasta pentru a continua...')
        # cv.imshow('image', np.uint8(image))
        # cv.waitKey(0)



