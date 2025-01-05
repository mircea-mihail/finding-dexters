import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

