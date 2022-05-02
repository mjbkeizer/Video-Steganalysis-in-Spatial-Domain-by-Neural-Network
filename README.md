# Video Steganalysis using Noise Residual Neural Network
#### Author: Mart Keizer

## Environment setup:
1. Create new Conda environment with Python 3.7.
2. Run the following command: 'pip install opencv-python==4.5.5.62 torch==1.10.2 torchvision==0.11.3 matplotlib==3.5.1 moviepy==1.0.3 scikit-image==0.19.2 h5py==2.10.0'
3. Run the following command: 'conda install tensorflow-gpu==2.1.0'

## Get VLOG dataset
1. Create ./VLOG_dataset folder. (Change PathSettings.py if you want to use a different directory)
2. Download <a href="https://web.eecs.umich.edu/~fouhey/2017/VLOG/agree.html" target="_blank">VLOG dataset</a> and put it in the VLOG_dataset folder.
3. The VLOG dataset videos are located in a bunch of subfolders. Follow the following steps to get all videos in the parent folder. (./VLOG_dataset)
   1. Go to the ./VLOG_dataset folder and press F3 to open the search dialog (or if you have Windows 7+ it will move the cursor to the search bar)
   2. Type in \*.\* and press enter. (This should find all files located in the subfolders)
   3. Wait for the search to complete. Note that it can appear to be done and then suddenly it finds more files. There is no notification when the search is complete other than a bar saying: Search again in, which appears at the bottom of the search results.
   4. Select all files using Ctrl + A
   5. Right click, choose cut
   6. Move to the ./VLOG_dataset folder, right-click an empty place and choose paste.
   7. You can remove all subfolders after you verrified that they are now empty.

## CREATE STEGANOGRAPHY DATASET:
1. Run VLOG_SplitVideos.py to create smaller (10 sec) videos out of the VLOG dataset videos.
2. Check the ./dataset/videos_10sec folder to make sure there are 10-second videos in there now.
3. Optional: delete VLOG dataset to free up some space on you hard drive.
4. Run Regular_CopyVideos.py and DeepStego_CreateVideos.py. Now we have regular videos and deep stego videos.
5. To get MSU Stego videos we need to concatenate 10-second videos. This way we only have to use the MSU StegoVideo tool on a few videos and not on all 10 secons videos seperate.
   1. Run MSU_ConcatenateVideos.py to concatenate videos. They will be located in the ./dataset/msu_stego_videos folder.
   2. Put the Secret.txt file in the video(s) using MSU StegoVideo. (Output should be .avi video(s))
   3. Run MSU_SlitVideos.py to split the videos again to 10 second videos. 
6. Now we have three categories of videos in 'msu_stego_videos', 'regular_videos' and deep_stego_videos '
7. Run CreateAnnotationFiles.py to extract frames from the videos and create the required annotation files needed for loading the training data.
8. Now we can start training the model!

## TRAIN THE MODEL:
1. Run TrainCNN.py to train the model.

## TEST AND EVALUATE THE MODEL:
1. Run GetResults.py to test and evaluate the model.
