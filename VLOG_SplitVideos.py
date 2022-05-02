import os
from HelperFunctions import split_videos
from Settings import *


print("Splitting videos...")

# check if VLOG folder exist
if not os.path.isdir(VLOG_dir):
    os.makedirs(VLOG_dir)

# check if directory for 10 seconds videos already exist, if not create it
if not os.path.isdir(dir_10sec):
    os.makedirs(dir_10sec)

# iterate over files in directory
i = 1
for file_name in os.listdir(VLOG_dir):
    file_path = os.path.join(VLOG_dir, file_name)
    # checking if it is a file
    if os.path.isfile(file_path):
        # checking if it is a mp4 file
        if file_path[-4:] == '.mp4':
            # split videos into 10-second videos
            split_videos(
                file_name=file_path,
                output_file_name='VLOG_video_' + str(i),
                output_path=dir_10sec,
                file_type="mp4",
                seconds=10
            )
            i += 1

# no videos where found in VLOG folder
if i == 1:
    print("ERROR: Download the VLOG dataset and put it in", VLOG_dir, "before you run this script!")
    exit(0)

print("Finished splitting videos!")
