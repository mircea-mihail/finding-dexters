import pandas as pd
import cv2 as cv

from constants import *

def get_character_rows(character):
    train_notation_files = [file for file in os.listdir(TRAIN_DIR) if os.path.splitext(file)[1] == ".txt"]
    rows_to_return = []

    for file in train_notation_files:
        df = pd.read_csv(os.path.join(TRAIN_DIR, file), header=None)

        for i in range(df.shape[0]): 
            row = (df.iloc[i]).tolist()[0].split(" ")
            if row[5] == character:
                for i in range(1, 5):
                    row[i] = int(row[i])
                row.append(file.split("_")[0])
                rows_to_return.append(row)
    return rows_to_return

def get_photo_rows(dir, photo):
    train_notation_files = [file for file in os.listdir(TRAIN_DIR) if os.path.splitext(file)[1] == ".txt" ]
    file_name = [file for file in train_notation_files if file.split("_")[0] == dir][0]

    df = pd.read_csv(os.path.join(TRAIN_DIR, file_name), header=None, delimiter=" ")
    filtered_df = df[df[0] == photo]
    rows_to_return = filtered_df.apply(lambda row: [row[0]] + [int(x) for x in row[1:5]] + [row[5]], axis=1).tolist()

    return rows_to_return

def get_width(row):
    return row[3] - row[1]

def get_height(row):
    return row[4] - row[2]

def read_photo(idx, dir):
    img = cv.imread(os.path.join(os.path.join(TRAIN_DIR, dir), f"{idx+1:04d}.jpg"))
    if img.shape != STD_IMG_SHAPE:
        img = cv.resize(img, (STD_IMG_SHAPE[1], STD_IMG_SHAPE[0]))
    return img

