import os
import random
from HelperFunctions import concatenate_videos
from Settings import *

method = "reduce"  # 'reduce' or 'compose'
size_output_video = 300  # number of videos to concatenate in 1 video

if __name__ == "__main__":
    # check if directory for MSU stego videos already exist, if not create it
    if not os.path.isdir(dir_msu_stego):
        os.makedirs(dir_msu_stego)

    # gather videos from folder in 'coverVideos' list
    cover_videos = []
    for file_name in os.listdir(dir_10sec):
        file_path = os.path.join(dir_10sec, file_name)
        # checking if it is a file
        if os.path.isfile(file_path):
            # checking if file type == .mp4
            if file_path[-4:] == '.mp4':
                cover_videos.append(file_path)

    # shuffle videos to generalise the dataset
    random.shuffle(cover_videos)
    # initialise variables
    x, i, j = 1, 0, size_output_video
    # check if number of videos in 1 file does not exceed 'class_category_size'
    if j > class_category_size:
        j = class_category_size

    print("Concatenating videos...")
    while j <= len(cover_videos) - 1:
        if len(cover_videos[i:j]) == 0:
            break
        # concatenate video i until j and put video in output folder
        concatenate_videos(cover_videos[i:j], dir_msu_stego + "/video_" + str(x) + ".mp4", method)
        # update for next iteration
        x += 1
        i += size_output_video
        # update j and check if exceed 'class_category_size'
        if j + size_output_video < class_category_size:
            j += size_output_video
        else:
            j = class_category_size

    print("Finished conatenating videos: " + int(x) + " videos can be found in " + dir_msu_stego)
