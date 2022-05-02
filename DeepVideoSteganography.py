import os
import numpy as np
import cv2
import math
import sys
from tensorflow.keras.models import load_model
from skimage.util.shape import view_as_blocks


# Normalize input images
def normalize_batch(imgs):
    return (imgs - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])


# Denormalize output images
def denormalize_batch(imgs, should_clip=True):
    imgs = (imgs * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])

    if should_clip:
        imgs = np.clip(imgs, 0, 1)
    return imgs


# Custom block shuffling
def shuffle(img, inverse=False):
    # Configure block size, rows and columns
    blk_size = 56
    rows = np.uint8(img.shape[0] / blk_size)
    cols = np.uint8(img.shape[1] / blk_size)

    # Create a block view on image
    img_blks = view_as_blocks(img, block_shape=(blk_size, blk_size, 3)).squeeze()
    img_shuff = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Secret key maps
    map = {0: 2, 1: 0, 2: 3, 3: 1}
    inv_map = {v: k for k, v in map.items()}

    # Perform block shuffling
    for i in range(0, rows):
        for j in range(0, cols):
            x, y = i * blk_size, j * blk_size
            if inverse:
                img_shuff[x:x + blk_size, y:y + blk_size] = img_blks[inv_map[i], inv_map[j]]
            else:
                img_shuff[x:x + blk_size, y:y + blk_size] = img_blks[map[i], map[j]]

    return img_shuff


# Update progress bar
def update_progress(current_frame, total_frames):
    progress = math.ceil((current_frame / total_frames) * 100)
    sys.stdout.write('\rProgress: [{0}] {1}%'.format('>' * math.ceil(progress / 10), progress))


def video_hide(model, secret_video, cover_video, output_video, shuffle_vid=False):
    # Load the trained model
    model = load_model(model, compile=False)

    # Input videos - Secret and Cover
    vidcap1 = cv2.VideoCapture(secret_video)
    vidcap2 = cv2.VideoCapture(cover_video)

    # Start video encoding
    print("\nEncoding video ...\n")

    # Total secret video frames
    num_frames = int(vidcap1.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames in secret video:", num_frames)

    # Video writer for output
    secret_video_name = secret_video.split('\\')[-1]
    secret_video_name = os.path.splitext(secret_video_name)[0]

    outputFileName = output_video + "\\" + cover_video.split('\\')[-1]
    outputFileName = os.path.splitext(outputFileName)[0] + ' (' + secret_video_name + ').avi'

    container_outvid = cv2.VideoWriter(outputFileName, cv2.VideoWriter_fourcc('H', 'F', 'Y', 'U'), 15, (224, 224))

    # Temporary buffers for batching
    secret_batch = []
    cover_batch = []
    frame = 0

    # Process frames as batches
    while True:

        # Read frames sequentially
        (success1, secret) = vidcap1.read()
        (success2, cover) = vidcap2.read()

        if not (success1 and success2):
            break

        # Preprocess frames
        secret = cv2.resize(cv2.cvtColor(secret, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA)
        cover = cv2.resize(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA)

        # Perform block shuffle
        if shuffle_vid:
            secret = shuffle(secret, inverse=False)

        # Append frames to buffer
        secret_batch.append(secret)
        cover_batch.append(cover)
        # print("Batching...")
        frame = frame + 1

        # Perform batch prediction
        if frame % 4 == 0:

            # Convert images to float type
            secret_batch = np.float32(secret_batch) / 255.0
            cover_batch = np.float32(cover_batch) / 255.0

            # Predict outputs
            coverout = model.predict([normalize_batch(secret_batch), normalize_batch(cover_batch)])

            # Postprocess cover image output
            coverout = denormalize_batch(coverout)
            coverout = np.squeeze(coverout) * 255.0
            coverout = np.uint8(coverout)

            # Save cover output video
            for i in range(0, 4):
                # imageio.imwrite("coverout.png",frame)
                container_outvid.write(coverout[i][..., ::-1])

            # Empty temporary buffers
            secret_batch = []
            cover_batch = []

            # Update progress
            update_progress(frame, num_frames)

    # Finish video encoding
    print("\n\nSuccessfully encoded video !!!\n")

    # Close video capturers
    vidcap1.release()
    vidcap2.release()
    cv2.destroyAllWindows()
