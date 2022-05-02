import os
import random
from Settings import *

if __name__ == "__main__":

    # check if regular video folder exist
    if not os.path.isdir(dir_regular):
        os.makedirs(dir_regular)

    regularVideos = []  # list of paths to videos

    # iterate over files in directory
    for file_name in os.listdir(dir_10sec):
        f = dir_10sec + '/' + file_name
        # checking if it is a file
        if os.path.isfile(f):
            if f[-4:] == '.mp4':
                regularVideos.append(f)

    # shuffle the videos
    random.shuffle(regularVideos)

    # copy the videos to use for training
    for i in range(class_category_size):
        regular_video_path = regularVideos[i]
        print(regular_video_path)
        os.rename(regular_video_path, dir_regular + "/" + regular_video_path.split('/')[-1])

        print("Progress: " + str(i) + "/" + str(class_category_size))

    print("Finished copying regular videos!")
