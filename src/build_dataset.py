import cv2 as cv
import random
import time
import numpy as np
import shutil

from general_utility import *
from character_classes import *

def build_dexter_positives():
    rows = get_character_rows("dexter")
    img_idx = 0
    for row in rows:
        img = cv.imread(os.path.join(os.path.join(TRAIN_DIR, row[6]), row[0]))
        width = get_width(row)
        height = int(width / dexter.xy_ratio)
        trimmed = img[row[2]:row[2]+height, row[1]:row[3]]
        downscaled = cv.resize(trimmed, (dexter.width, dexter.height))
        cv.imwrite(os.path.join(os.path.join(FACES_DIR, dexter.face_dir), f"{img_idx:04d}.png"), downscaled)
        img_idx += 1

def build_deedee_positives():
    rows = get_character_rows("deedee")
    img_idx = 0
    for row in rows:
        img = cv.imread(os.path.join(os.path.join(TRAIN_DIR, row[6]), row[0]))
        width = get_width(row)
        height = int(width / deedee.xy_ratio)
        trimmed = img[row[2]:row[2]+height, row[1]:row[3]]
        downscaled = cv.resize(trimmed, (deedee.width, deedee.height))
        cv.imwrite(os.path.join(os.path.join(FACES_DIR, deedee.face_dir), f"{img_idx:04d}.png"), downscaled)
        img_idx += 1

def build_dad_positives():
    rows = get_character_rows("dad")
    img_idx = 0
    for row in rows:
        img = cv.imread(os.path.join(os.path.join(TRAIN_DIR, row[6]), row[0]))
        width = get_width(row)
        height = int(width / dad.xy_ratio)
        trimmed = img[row[2]:row[2]+height, row[1]:row[3]]
        downscaled = cv.resize(trimmed, (dad.width, dad.height))
        cv.imwrite(os.path.join(os.path.join(FACES_DIR, dad.face_dir), f"{img_idx:04d}.png"), downscaled)
        img_idx += 1

def build_mom_positives():
    rows = get_character_rows("mom")
    img_idx = 0
    for row in rows:
        img = cv.imread(os.path.join(os.path.join(TRAIN_DIR, row[6]), row[0]))
        width = get_width(row)
        height = int(width / mom.xy_ratio)
        trimmed = img[row[2]:row[2]+height, row[1]:row[3]]
        downscaled = cv.resize(trimmed, (mom.width, mom.height))
        cv.imwrite(os.path.join(os.path.join(FACES_DIR, mom.face_dir), f"{img_idx:04d}.png"), downscaled)
        img_idx += 1

def build_unknown_positives():
    rows = get_character_rows("unknown")
    img_idx = 0
    for row in rows:
        img = cv.imread(os.path.join(os.path.join(TRAIN_DIR, row[6]), row[0]))
        width = get_width(row)
        height = int(width / unknown.xy_ratio)
        trimmed = img[row[2]:row[2]+height, row[1]:row[3]]
        downscaled = cv.resize(trimmed, (unknown.width, unknown.height))
        cv.imwrite(os.path.join(os.path.join(FACES_DIR, unknown.face_dir), f"{img_idx:04d}.png"), downscaled)
        img_idx += 1

def build_positives(descriptors, widths, heights, character_dir_name):
    img_idx = {}
    for i in range(len(descriptors)):
        img_idx[f"{widths[i]}X{heights[i]}"] = 0
    
    if character_dir_name == 'all':
        characters = CHARACTERS
    else:
        characters = [character_dir_name]

    for character in characters:
        rows = get_character_rows(character)
        for row in rows:
            img = cv.imread(os.path.join(os.path.join(TRAIN_DIR, row[6]), row[0]))

            width = get_width(row)
            height = get_height(row)
            ratio = width/height

            smallest_error = 10000
            smallest_err_idx = -1
            for i in range(len(descriptors)):
                current_error = abs(widths[i]/heights[i] - ratio) 
                if current_error < smallest_error:
                    smallest_error = current_error
                    smallest_err_idx = i
            
            height = int(width * heights[smallest_err_idx] / widths[smallest_err_idx])
            ratio_name = f"{widths[smallest_err_idx]}X{heights[smallest_err_idx]}"

            trimmed = img[row[2]:row[2]+height, row[1]:row[3]]
            downscaled = cv.resize(trimmed, 
                                   (widths[smallest_err_idx] * descriptors[smallest_err_idx],
                                    heights[smallest_err_idx] * descriptors[smallest_err_idx]
                                    ))
            faces_dir = os.path.join(FACES_DIR, character_dir_name)
            current_size_dir = os.path.join(faces_dir, ratio_name)

            if not os.path.exists(faces_dir):
                os.makedirs(faces_dir)
            if not os.path.exists(current_size_dir):
                os.makedirs(current_size_dir)
            
            cv.imwrite(os.path.join(current_size_dir,f"{img_idx[ratio_name]:04d}.png"), downscaled)
            img_idx[ratio_name] += 1

def get_small_dataset():
    file_nr=  0
    files = os.listdir(os.path.join(FACES_DIR, all.face_dir))
    files.sort()
    for i in range(0, len(files), 60):
        shutil.copy(os.path.join(os.path.join(FACES_DIR, all.face_dir), files[i]), os.path.join(SMALL_SET_DIR, files[i]))

def generate_negative_rectangle(rows, character_dir, min_variance):
    random.seed(time.time())

    img = cv.imread(os.path.join(os.path.join(TRAIN_DIR, character_dir), rows[0][0]))
    for _ in range(RANDOM_IMG_GEN_TRIES):
        x1 = random.randint(0, STD_IMG_SHAPE[1] - AVG_FACE_WIDTH)
        y1 = random.randint(0, STD_IMG_SHAPE[0] - AVG_FACE_HEIGHT)
        x2 = x1 + AVG_FACE_WIDTH 
        y2 = y1 + AVG_FACE_HEIGHT
        no_intersection = False
        rect_points = (x1, y1), (x2, y2)

        neg_ex = img[rect_points[0][1]:rect_points[1][1], rect_points[0][0]:rect_points[1][0]]
        if(np.var(neg_ex) < min_variance):
            continue

        for row in rows:
            gx1 = row[1]
            gy1 = row[2]
            gx2 = row[3]
            gy2 = row[4]

            no_intersection = (x2 <= gx1 or  # New rectangle is completely to the left
                            x1 >= gx2 or  # New rectangle is completely to the right
                            y2 <= gy1 or  # New rectangle is completely above
                            y1 >= gy2)    # New rectangle is completely below

            if not(no_intersection):
                break

        if not(no_intersection):
            continue

        return (x1, y1), (x2, y2)

    return (0, 0), (0, 0)

def build_negatives():
    success = 0
    all = 0
    variances_to_get = np.arange(0, AVG_FACE_VARIANCE * 3, AVG_FACE_VARIANCE * 3 / 12)
    print(variances_to_get)

    for min_variance in variances_to_get:
        for character in CHARACTERS[:4]:
            for i in range(NR_CHARACTER_PHOTOS):
                rows = get_photo_rows(character, f"{i+1:04d}.jpg")
                all += 1
                img = cv.imread(os.path.join(os.path.join(TRAIN_DIR, character), rows[0][0]))
                rect_points = generate_negative_rectangle(rows, character, min_variance)
                if(rect_points != ((0, 0), (0, 0))):
                    neg_ex = img[rect_points[0][1]:rect_points[1][1], rect_points[0][0]:rect_points[1][0]]
                    cv.imwrite(os.path.join(NEGATIVE_DIR, f"{success:04d}.png"), neg_ex)
                    success += 1
        min_variance += LOWEST_FACE_VARIANCE

        print(f"accepted ratio: {success / all}, successes: {success}")
