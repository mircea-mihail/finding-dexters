class dexter:
    xy_ratio = 1.1458
    width = 48
    height = 42

    width_hog_cell = 8
    height_hog_cell = 7
    
    face_dir = "dexter"

class deedee:
    xy_ratio = 1.35416
    descriptors = 5
    width = 55
    height = 40

    width_hog_cell = 11
    height_hog_cell = 8
    
    face_dir = "deedee"


class dad:
    xy_ratio = 0.729166

    width_hog_cell = 8
    height_hog_cell = 11
    
    descriptors = 5
    width = width_hog_cell * descriptors
    height = height_hog_cell * descriptors
   
    face_dir = "dad"

class mom:
    xy_ratio = 0.9375

    width_hog_cell = 6
    height_hog_cell = 6
    
    descriptors = 6
    width = width_hog_cell * descriptors
    height = height_hog_cell * descriptors
   
    face_dir = "mom"

class unknown:
    xy_ratio = 1

    width_hog_cell = 6
    height_hog_cell = 6
    
    descriptors = 6
    width = width_hog_cell * descriptors
    height = height_hog_cell * descriptors
   
    face_dir = "unknown"

class all:
    xy_ratio = 1

    width_hog_cell = 6
    height_hog_cell = 6
    
    descriptors = 6
    width = width_hog_cell * descriptors
    height = height_hog_cell * descriptors
   
    face_dir = "all"


