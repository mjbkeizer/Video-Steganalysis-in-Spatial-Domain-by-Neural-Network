import os
import random
from HelperFunctions import count_frames_video, extract_frames_for_training
from Settings import *


def random_reorder_annotation_file(annotation_file_name):
    with open(dir_CNN_dataset + annotation_file_name, 'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()
    with open(dir_CNN_dataset + annotation_file_name, 'w') as target:
        for _, line in data:
            target.write(line)


if __name__ == "__main__":

    print("Extracting frames from video...")
    extract_frames_for_training(dir_deep_stego, dir_class_deep_stego)
    extract_frames_for_training(dir_regular, dir_class_regular)
    extract_frames_for_training(dir_msu_stego, dir_class_msu_stego)

    print("Creating annotation files...")
    for class_dir_name in os.listdir(dir_CNN_dataset):
        class_dir = os.path.join(dir_CNN_dataset, class_dir_name)

        if not os.path.isfile(class_dir):
            for videoFolderName in os.listdir(class_dir):
                video_dir = os.path.join(class_dir, videoFolderName)

                # check if this is a folder
                if not os.path.isfile(video_dir):
                    numberOfFrames = count_frames_video(video_dir)
                    if class_dir_name == 'regular':
                        classIndex = "0"
                    elif class_dir_name == 'msu_stego':
                        classIndex = "1"
                    elif class_dir_name == "deep_stego":
                        classIndex = "2"
                    else:
                        raise ValueError("Directories in " + dir_CNN_dataset + " should be 'regular', 'msu_stego' or 'deep_stego'!")

                    choices = [train_annotations, test_annotations, validate_annotations]
                    choice_annotation = random.choices(choices, weights=[10, 10, 1], k=1)[0]

                    # Open the file in append & read mode ('a+')
                    with open(dir_CNN_dataset + choice_annotation, "a+") as file_object:
                        # Move read cursor to the start of file.
                        file_object.seek(0)
                        # If file is not empty then append '\n'
                        data = file_object.read(100)
                        if len(data) > 0:
                            file_object.write("\n")
                        # Append text at the end of file
                        file_object.write(
                            '"' + class_dir_name + '/' + videoFolderName + '"' + ' 1 ' + str(
                                numberOfFrames) + ' ' + classIndex)

    # randomly change order of videos in annotation files, better for training network
    random_reorder_annotation_file(train_annotations)
    random_reorder_annotation_file(test_annotations)
    random_reorder_annotation_file(validate_annotations)

    print("Finished creating annotation files")