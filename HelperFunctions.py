import math
import cv2
import os
import torch
import shutil
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips
from mpl_toolkits.axes_grid1 import ImageGrid
from VideoFrameDataset import VideoFrameDataset


def concatenate_videos(video_clip_paths, output_path, method="compose"):
    """
    Concatenates several video files into one video file
    and save it to `output_path`. Note that extension (mp4, etc.) must be added to `output_path`
    `method` can be either 'compose' or 'reduce':
        `reduce`: Reduce the quality of the video to the lowest quality on the list of `video_clip_paths`.
        `compose`: type help(concatenate_videoclips) for the info
    """
    # create VideoFileClip object for each video file
    clips = [VideoFileClip(c) for c in video_clip_paths]
    if method == "reduce":
        # calculate minimum width & height across all clips
        min_height = min([c.h for c in clips])
        min_width = min([c.w for c in clips])
        # resize the videos to the minimum
        clips = [c.resize(newsize=(min_width, min_height)) for c in clips]
        # concatenate the final video
        final_clip = concatenate_videoclips(clips)
    elif method == "compose":
        # concatenate the final video with the compose method provided by moviepy
        final_clip = concatenate_videoclips(clips, method="compose")
    # write the output video file
    final_clip.write_videofile(output_path, fps=24, threads=5, codec="libx264")


def get_duration_video(file_name):
    """
    Returns duration of a video in seconds.
    file_name: path to video file
    """
    video = cv2.VideoCapture(file_name)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    return frame_count / fps


def split_videos(file_name, output_file_name, output_path, file_type, seconds):
    """
    Split a video into videos of each a specified number of seconds.
    file_name: path to original video file
    output_file_name: name of output videos
    output_path: path to output directory
    file_type: file type of output videos
    seconds: the length in seconds of the splitted videos
    """
    # get rounded-down duration of video in seconds
    duration = get_duration_video(file_name)
    rounded_duration = math.floor(duration / 10) * 10
    # initialise variables
    start_time = 0
    end_time = seconds
    i = 1
    while end_time <= rounded_duration:
        # extract part of the video
        ffmpeg_extract_subclip(
            filename=file_name,
            t1=start_time,
            t2=end_time,
            targetname=str(output_path) + "/" + str(output_file_name) + "_" + str(i) + "." + file_type
        )
        # update variables
        start_time += seconds
        end_time += seconds
        i = i + 1


def extract_frames_for_training(input_dir, output_dir):
    """
    Extracting frames from video directory to CNN dataset directory
    to be used by VideoFrameDataset.
    input_dir: the path to the directory with the videos
    output_dir: the path to the directory of the dataset
    """
    # iterate over files in directory
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        # checking if it is a file
        if os.path.isfile(file_path):
            # capture the video
            vid_cap = cv2.VideoCapture(file_path)
            video_name = file_path.split('/')[-1]
            video_name = os.path.splitext(video_name)[0]
            # create directory for frames
            if not os.path.isdir(output_dir + video_name):
                os.makedirs(output_dir + video_name)
            # read video file
            success, image = vid_cap.read()
            count = 1
            while success:
                # save frame as JPEG file
                newFilePath = output_dir + video_name + '/' + 'img_' + f'{count:05}' + '.jpg'
                cv2.imwrite(newFilePath, image)
                success, image = vid_cap.read()
                count += 1
            vid_cap.release()
            os.remove(file_path)  # remove full video


def count_frames_video(video_path):
    """
    Returns the number of frames from a video.
    video_path: the path to the video file
    """
    initial_count = 0
    for path in os.listdir(video_path):
        if os.path.isfile(os.path.join(video_path, path)):
            initial_count += 1
    return initial_count


def plot_video(rows, cols, frame_list, plot_width, plot_height, title: str):
    transform = transforms.ToPILImage()
    new_frame_list = []
    for x in frame_list:
        new_frame_list.append(transform(x))
    frame_list = new_frame_list
    fig = plt.figure(figsize=(plot_width, plot_height))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )
    for index, (ax, im) in enumerate(zip(grid, frame_list)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(index)
    plt.suptitle(title)
    plt.show()


def check_cuda():
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available, good job!")
        return True
    else:
        print("CUDA is not available!")
        print("Torch version: " + torch.__version__)
        try:
            torch.rand(1, device="cuda")  # Use to get specific error message
        except Exception as e:
            print('You might get more debug info here: ' + str(e))
        return False


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


def get_frame_dataset(root, annotation_file, segments, frames, pre_process):
    return VideoFrameDataset(
        root_path=root,
        annotationfile_path=annotation_file,
        num_segments=segments,
        frames_per_segment=frames,
        imagefile_template='img_{:05d}.jpg',
        transform=pre_process,
        test_mode=False
    )

