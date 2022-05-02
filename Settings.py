total_size_dataset = 300  # number of 10sec videos in dataset
class_category_size = round(total_size_dataset / 3)

VLOG_dir = './VLOG_dataset'  # change this to your VLOG directory
dataset_dir = './dataset'  # change this to where you want to store you dataset

dir_10sec = dataset_dir + '/videos_10sec'
dir_deep_stego = dataset_dir + '/deep_stego_videos'
dir_msu_stego = dataset_dir + '/msu_stego_videos'

dir_regular = dataset_dir + '/regular_videos'
dir_secret = dataset_dir + '/secret_videos'
dir_original = dataset_dir + '/original_videos'

dir_CNN_dataset = dataset_dir + '/CNN_dataset'
dir_class_deep_stego = dir_CNN_dataset + '/deep_stego/'
dir_class_msu_stego = dir_CNN_dataset + '/msu_stego/'
dir_class_regular = dir_CNN_dataset + '/regular/'

train_annotations = 'trainAnnotations.txt'
validate_annotations = 'validateAnnotations.txt'
test_annotations = 'testAnnotations.txt'


