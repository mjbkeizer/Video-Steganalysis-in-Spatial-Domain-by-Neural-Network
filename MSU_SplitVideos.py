import os
from HelperFunctions import split_videos
from Settings import *

if __name__ == "__main__":
    # check if directory for MSU stego videos already exist, if not create it
    if not os.path.isdir(dir_msu_stego):
        os.makedirs(dir_msu_stego)

    # iterate over files in directory
    for filename in os.listdir(dir_msu_stego):
        f = os.path.join(dir_msu_stego, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if f[-4:] == '.avi':
                filename = filename.replace(f[-4:], '')
                split_videos(f, filename, dir_msu_stego, "avi", 10)
