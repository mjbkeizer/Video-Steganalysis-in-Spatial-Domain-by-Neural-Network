import os
import random
# Sometimes the following line is needed if you get Error: "Not creating XLA devices, tf_xla_enable_xla_devices not set"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
from DeepVideoSteganography import video_hide
from Settings import *

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    if len(tf.config.experimental.list_physical_devices('GPU')) < 1:
        print("It is advised to use a GPU with Cuda! Without a GPU this will take a long time.")

    # check if regular video folder exist
    if not os.path.isdir(dir_deep_stego):
        os.makedirs(dir_deep_stego)

    # iterate over files in directory
    videos = []
    for filename in os.listdir(dir_10sec):
        file_path = os.path.join(dir_10sec, filename)
        # check if it is a file
        if os.path.isfile(file_path):
            # check if file is video
            if file_path[-4:] == '.mp4':
                videos.append(file_path)

    # shuffle the videos
    random.shuffle(videos)

    # divide videos to cover and secret list
    coverVideos = videos[:len(videos)//2]
    secretVideos = videos[len(videos)//2:]

    # create a specified number of deep stego videos (can be adjusted in Settings.py)
    for i in range(class_category_size):
        secret_video_path = secretVideos[i]
        cover_video_path = coverVideos[i]

        video_hide('stego_model/hide.h5', secret_video_path, cover_video_path, dir_deep_stego, True)

        os.remove(secret_video_path)
        os.remove(cover_video_path)

        print("Progress: " + str(i+1) + "/" + str(class_category_size) + " videos")

    print("Finished creating Deep Video Steganography videos")
